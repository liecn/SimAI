#include "Topology.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <set>
#include <cmath> // Required for std::isfinite
#include <algorithm> // Required for std::find

std::shared_ptr<EventQueue> Topology::event_queue = nullptr;

void Topology::set_event_queue(std::shared_ptr<EventQueue> event_queue) noexcept {
    assert(event_queue != nullptr);
    Topology::event_queue = std::move(event_queue);
}

Topology::Topology(int devices_count, int npus_count) noexcept 
    : npus_count(-1), devices_count(-1), dims_count(-1), 
      last_batch_time_(0), batch_timeout_event_id_(0) {
    npus_count_per_dim = {};
    this->devices_count = devices_count;
    this->npus_count = npus_count;
    for (int i = 0; i < devices_count; ++i) {
        devices.push_back(std::make_shared<Device>(i));
    }
}

int Topology::get_devices_count() const noexcept {
    assert(devices_count > 0);
    assert(npus_count > 0);
    assert(devices_count >= npus_count);
    return devices_count;
}

int Topology::get_npus_count() const noexcept {
    assert(devices_count > 0);
    assert(npus_count > 0);
    assert(devices_count >= npus_count);
    return npus_count;
}

int Topology::get_dims_count() const noexcept {
    assert(dims_count > 0);
    return dims_count;
}

std::vector<int> Topology::get_npus_count_per_dim() const noexcept {
    assert(npus_count_per_dim.size() == dims_count);
    return npus_count_per_dim;
}

std::vector<Bandwidth> Topology::get_bandwidth_per_dim() const noexcept {
    assert(bandwidth_per_dim.size() == dims_count);
    return bandwidth_per_dim;
}

void Topology::send(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);

    // Store chunk ownership
    active_chunks_ptrs.push_back(std::move(chunk));
    Chunk* chunk_ptr = active_chunks_ptrs.back().get();
    
    // Set topology reference and start time
    chunk_ptr->set_topology(this);
    chunk_ptr->set_transmission_start_time(event_queue->get_current_time());
    
    // Add to simulation (this will calculate rates and schedule completion)
    add_chunk_to_links(chunk_ptr);
}

void Topology::send_with_batching(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);
    
    const auto current_time = event_queue->get_current_time();
    
    // Store chunk ownership
    active_chunks_ptrs.push_back(std::move(chunk));
    Chunk* chunk_ptr = active_chunks_ptrs.back().get();
    
    // Add to pending batch
    pending_chunks_.push_back(chunk_ptr);
    
    // If this is the first chunk in a potential batch, schedule timeout
    if (pending_chunks_.size() == 1) {
        last_batch_time_ = current_time;
        
        // Cancel any existing batch timeout event
        if (batch_timeout_event_id_ > 0) {
            event_queue->cancel_event(batch_timeout_event_id_);
        }
        
        batch_timeout_event_id_ = event_queue->schedule_event(
            current_time + BATCH_TIMEOUT_NS, 
            batch_timeout_callback, 
            this
        );
    }
    
    // Check if we should process immediately (e.g., large batch or timeout)
    // For now, rely on timeout - could add smarter heuristics later
}

void Topology::connect(DeviceId src, DeviceId dest, Bandwidth bandwidth, Latency latency, bool bidirectional) noexcept {
    assert(0 <= src && src < devices_count);
    assert(0 <= dest && dest < devices_count);
    assert(bandwidth > 0);
    assert(latency >= 0);

    auto link = std::make_shared<Link>(bandwidth, latency);
    link_map[std::make_pair(src, dest)] = link;

    if (bidirectional) {
        auto reverse_link = std::make_shared<Link>(bandwidth, latency);
        link_map[std::make_pair(dest, src)] = reverse_link;
    }
}

std::shared_ptr<Device> Topology::get_device(int index) {
    return this->devices.at(index);
}

// Proven chunk-based simulation methods (from flowsim-old)

void Topology::update_link_states() {
    std::set<Chunk*> fixed_chunks;
    while (fixed_chunks.size() < active_chunks.size()) {
        double bottleneck_rate = std::numeric_limits<double>::max();
        std::pair<DeviceId, DeviceId> bottleneck_link;
        for (const auto& link : active_links) {
            double fair_rate = calculate_bottleneck_rate(link, fixed_chunks);
            if (fair_rate < bottleneck_rate) {
                bottleneck_rate = fair_rate;
                bottleneck_link = link;
            }
        }
        if (bottleneck_rate < std::numeric_limits<double>::max()) {
            for (Chunk* chunk : link_map[bottleneck_link]->active_chunks) {
                if (fixed_chunks.find(chunk) == fixed_chunks.end()) {
                    chunk->set_rate(bottleneck_rate);
                    fixed_chunks.insert(chunk);
                }
            }
        } else {
            break;
        }
    }
}

double Topology::calculate_bottleneck_rate(const std::pair<DeviceId, DeviceId>& link, const std::set<Chunk*>& fixed_chunks) {
    double remaining_bandwidth = link_map[link]->get_bandwidth();
    int active_chunks = 0;

    for (Chunk* chunk : link_map[link]->active_chunks) {
        if (fixed_chunks.find(chunk) == fixed_chunks.end()) {
            ++active_chunks;
        } else {
            remaining_bandwidth -= chunk->get_rate();
        }
    }

    double fair_rate = active_chunks > 0 ? remaining_bandwidth / active_chunks : std::numeric_limits<double>::max();
    return fair_rate;
}

double Topology::calculate_path_latency(Chunk* chunk) {
    const auto& route = chunk->get_route();
    double total_latency = 0.0;
    
    auto it = route.begin();
    while (it != route.end()) {
        auto src_device = (*it)->get_id();
        ++it;
        if (it == route.end()) {
            break;
        }
        auto dest_device = (*it)->get_id();
        
        auto link_key = std::make_pair(src_device, dest_device);
        auto link_it = link_map.find(link_key);
        
        if (link_it != link_map.end()) {
            total_latency += link_it->second->get_latency();
        }
    }
    
    return total_latency;
}

void Topology::schedule_next_completion() {
    const auto current_time = event_queue->get_current_time();

    // Find the flow that will complete first
    uint64_t earliest_completion_time = UINT64_MAX;
    Chunk* next_chunk = nullptr;

    for (Chunk* chunk : active_chunks) {
        double remaining_size = chunk->get_remaining_size();
        double new_rate = chunk->get_rate();
        
        // Prevent division by zero
        if (new_rate <= 0) {
            new_rate = 1.0;
        }
        
        // Include both transmission time and path latency
        double transmission_time = remaining_size / new_rate;
        double path_latency = calculate_path_latency(chunk);
        uint64_t completion_time = current_time + std::max(1.0, transmission_time + path_latency);

        if (completion_time < earliest_completion_time) {
            earliest_completion_time = completion_time;
            next_chunk = chunk;
        }
    }

    // Schedule the next completion event
    if (next_chunk != nullptr) {
        auto* chunk_ptr = static_cast<void*>(next_chunk);
        int new_event_id = event_queue->schedule_event(earliest_completion_time, chunk_completion_callback, chunk_ptr);
        next_chunk->set_completion_event_id(new_event_id);
    }
}

void Topology::add_chunk_to_links(Chunk* chunk) {
    // Set topology reference for callback
    chunk->set_topology(this);
    
    // Add to active chunks list
    active_chunks.push_back(chunk);
    
    // Associate chunk with links
    associate_chunk_with_links(chunk);
    
    // Set initial transmission start time
    chunk->set_transmission_start_time(event_queue->get_current_time());
    
    // Calculate rates for all chunks and schedule completion for this new chunk
    // This is more efficient than canceling all events
    update_link_states();
    
    // Schedule completion for the new chunk based on its calculated rate
    const auto current_time = event_queue->get_current_time();
    double remaining_size = chunk->get_remaining_size();
    double rate = chunk->get_rate();
    
    // Prevent division by zero
    if (rate <= 0) {
        rate = 1.0;
    }
    
    // Include both transmission time and path latency
    double transmission_time = remaining_size / rate;
    double path_latency = calculate_path_latency(chunk);
    uint64_t completion_time = current_time + std::max(1.0, transmission_time + path_latency);
    
    // Schedule completion event for this chunk
    auto* chunk_ptr = static_cast<void*>(chunk);
    int event_id = event_queue->schedule_event(completion_time, chunk_completion_callback, chunk_ptr);
    chunk->set_completion_event_id(event_id);
}

void Topology::associate_chunk_with_links(Chunk* chunk) {
    const auto& route = chunk->get_route();
    int hop_count = 0;
    auto it = route.begin();
    
    while (it != route.end()) {
        auto src_device = (*it)->get_id();
        ++it;
        if (it == route.end()) {
            break;
        }
        auto dest_device = (*it)->get_id();
        auto link_key = std::make_pair(src_device, dest_device);
        
        // Add chunk to link's active chunks
        if (link_map.find(link_key) != link_map.end()) {
            link_map[link_key]->active_chunks.push_back(chunk);
            active_links.insert(link_key);
            
            // Track link for efficient removal
            chunk->add_active_link_key(link_key);
        }
        
        hop_count++;
    }
}

void Topology::remove_chunk_from_links(Chunk* chunk) {
    // Use the recorded link keys for efficient O(k) removal where k is the number of links the chunk is on
    const auto& link_keys = chunk->get_active_link_keys();
    
    for (const auto& link_key : link_keys) {
        auto link_it = link_map.find(link_key);
        if (link_it != link_map.end() && link_it->second != nullptr) {
            auto& active_chunks = link_it->second->active_chunks;
            active_chunks.remove(chunk);
            
            // If link is now empty, remove from active_links
            if (active_chunks.empty()) {
                active_links.erase(link_key);
            }
        }
    }
}

void Topology::chunk_completion_callback(void* arg) noexcept {
    Chunk* chunk = static_cast<Chunk*>(arg);
    Topology* topology = chunk->get_topology();

    // 1. Invoke the chunk's callback first (notify ASTRA-Sim)
    chunk->invoke_callback();

    // 2. Remove this chunk from the simulation
    topology->remove_chunk_from_links(chunk);
    topology->active_chunks.remove(chunk);

    // 3. If there are still active chunks, update their state
    if (!topology->active_chunks.empty()) {
        const auto current_time = topology->event_queue->get_current_time();
        
        // Update remaining sizes for all active chunks
        for (Chunk* active_chunk : topology->active_chunks) {
            double elapsed_time = current_time - active_chunk->get_transmission_start_time();
            double transmitted_size = elapsed_time * active_chunk->get_rate();
            active_chunk->update_remaining_size(transmitted_size);
            active_chunk->set_transmission_start_time(current_time);
            
            // Cancel old completion event
            if (active_chunk->get_completion_event_id() > 0) {
                topology->event_queue->cancel_event(active_chunk->get_completion_event_id());
                active_chunk->set_completion_event_id(0);
            }
        }
        
        // Recalculate rates for all remaining chunks
        topology->update_link_states();
        
        // Reschedule completion times for all remaining chunks
        for (Chunk* active_chunk : topology->active_chunks) {
            double remaining_size = active_chunk->get_remaining_size();
            double new_rate = active_chunk->get_rate();
            
            // Prevent division by zero
            if (new_rate <= 0) {
                new_rate = 1.0;
            }
            
            // Calculate new completion time
            double transmission_time = remaining_size / new_rate;
            double path_latency = topology->calculate_path_latency(active_chunk);
            uint64_t completion_time = current_time + std::max(1.0, transmission_time + path_latency);
            
            // Schedule new completion event
            auto* chunk_ptr = static_cast<void*>(active_chunk);
            int event_id = topology->event_queue->schedule_event(completion_time, chunk_completion_callback, chunk_ptr);
            active_chunk->set_completion_event_id(event_id);
        }
    }
}

void Topology::cancel_all_events() noexcept {
    const auto current_time = event_queue->get_current_time();
    for (Chunk* chunk : active_chunks) {
        double elapsed_time = current_time - chunk->get_transmission_start_time();
        double transmitted_size = elapsed_time * chunk->get_rate();
        chunk->update_remaining_size(transmitted_size);
        event_queue->cancel_event(chunk->get_completion_event_id());
        chunk->set_completion_event_id(0);
    }
}

void Topology::reschedule_active_chunks() {
    const auto current_time = event_queue->get_current_time();
    uint64_t min_completion_time = UINT64_MAX;
    std::vector<Chunk*> next_chunks;
    
    for (Chunk* chunk : active_chunks) {
        double remaining_size = chunk->get_remaining_size();
        double new_rate = chunk->get_rate();
        double completion_time_delta = std::max(1.0, remaining_size / new_rate);
        uint64_t completion_time = current_time + completion_time_delta;
        
        chunk->set_transmission_start_time(current_time);
        chunk->set_remaining_size(remaining_size);
        
        if (completion_time < min_completion_time) {
            next_chunks.clear();
            min_completion_time = completion_time;
            next_chunks.push_back(chunk);
        } else if (completion_time == min_completion_time) {
            next_chunks.push_back(chunk);
        }
    }
    
    // Schedule all chunks that complete at the same time
    for (Chunk* chunk : next_chunks) {
        auto* chunk_ptr = static_cast<void*>(chunk);
        int new_event_id = event_queue->schedule_event(min_completion_time, chunk_completion_callback, chunk_ptr);
        chunk->set_completion_event_id(new_event_id);
    }
}

// Batching Implementation (Performance Optimization)

void Topology::process_batch_of_chunks() {
    if (pending_chunks_.empty()) {
        return;
    }
    
    // Reset the batch timeout event ID since the timeout has fired
    batch_timeout_event_id_ = 0;
    
    // Process all pending chunks as a batch - NO cancel_all_events() to avoid segfaults
    const auto current_time = event_queue->get_current_time();
    
    // Add all pending chunks to the simulation at once
    for (Chunk* chunk : pending_chunks_) {
        // Set topology reference and start time
        chunk->set_topology(this);
        chunk->set_transmission_start_time(current_time);
        
        // Add to active chunks list
        active_chunks.push_back(chunk);
        
        // Associate with links
        associate_chunk_with_links(chunk);
    }
    
    // Calculate rates for all chunks ONCE for the entire batch
    update_link_states();
    
    // Schedule completion events for all new chunks
    for (Chunk* chunk : pending_chunks_) {
        double remaining_size = chunk->get_remaining_size();
        double rate = chunk->get_rate();
        
        // Prevent division by zero
        if (rate <= 0) {
            rate = 1.0;
        }
        
        // Calculate completion time
        double transmission_time = remaining_size / rate;
        double path_latency = calculate_path_latency(chunk);
        uint64_t completion_time = current_time + std::max(1.0, transmission_time + path_latency);
        
        // Schedule completion event
        auto* chunk_ptr = static_cast<void*>(chunk);
        int event_id = event_queue->schedule_event(completion_time, chunk_completion_callback, chunk_ptr);
        chunk->set_completion_event_id(event_id);
    }
    
    // Clear the batch
    pending_chunks_.clear();
}

void Topology::batch_timeout_callback(void* arg) noexcept {
    Topology* topology = static_cast<Topology*>(arg);
    topology->process_batch_of_chunks();
}
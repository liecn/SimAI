#include "Topology.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <set>
#include <cmath> // Required for std::isfinite

std::shared_ptr<EventQueue> Topology::event_queue = nullptr;

void Topology::set_event_queue(std::shared_ptr<EventQueue> event_queue) noexcept {
    assert(event_queue != nullptr);
    Topology::event_queue = std::move(event_queue);
}

Topology::Topology(int devices_count, int npus_count) noexcept : npus_count(-1), devices_count(-1), dims_count(-1) {
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

    chunk->set_topology(this);
    cancel_all_events();

    active_chunks_ptrs.push_back(std::move(chunk));
    Chunk* chunk_ptr = active_chunks_ptrs.back().get();
    active_chunks.push_back(chunk_ptr);

    add_chunk_to_links(chunk_ptr);
    update_link_states();
    reschedule_active_chunks();
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

    // std::cerr << "Debug: Connecting src: " << src << " dest: " << dest << " with bandwidth: " << bandwidth << std::endl;
    // if (bidirectional) {
    //     std::cerr << "Debug: Connecting dest: " << dest << " src: " << src << " with bandwidth: " << bandwidth << std::endl;
    // }
}

std::shared_ptr<Device> Topology::get_device(int index) {
    return this->devices.at(index);
}

void Topology::update_link_states() {
    // Simple O(1) per-flow rate calculation - MUCH faster!
    for (Chunk* chunk : active_chunks) {
        double bottleneck_rate = std::numeric_limits<double>::max();
        
        // Find bottleneck rate along the flow's path
        const auto& route = chunk->get_route();
        auto it = route.begin();
        while (it != route.end()) {
            auto src_device = (*it)->get_id();
            ++it;
            if (it == route.end()) break;
            auto dest_device = (*it)->get_id();
            
            auto link_key = std::make_pair(src_device, dest_device);
            if (link_map.find(link_key) != link_map.end()) {
                double link_bandwidth = link_map[link_key]->get_bandwidth();
                int active_flows_on_link = link_map[link_key]->active_chunks.size();
                
                if (active_flows_on_link > 0) {
                    double fair_share = link_bandwidth / active_flows_on_link;
                    bottleneck_rate = std::min(bottleneck_rate, fair_share);
                }
            }
        }
        
        // Set the bottleneck rate for this flow
        if (bottleneck_rate < std::numeric_limits<double>::max()) {
            chunk->set_rate(bottleneck_rate);
        } else {
            chunk->set_rate(1.0); // Minimum rate
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
    
    // Ensure fair_rate is positive and finite
    if (fair_rate <= 0 || !std::isfinite(fair_rate)) {
        fair_rate = 1.0; // Set a minimum rate
    }
    
    static int debug_rate_count = 0;
    debug_rate_count++;
    if (debug_rate_count <= 10) {
        std::cout << "[RATE] Link (" << link.first << " -> " << link.second 
                  << "): bandwidth=" << link_map[link]->get_bandwidth() << " bytes/ns, "
                  << "active_chunks=" << active_chunks << ", fair_rate=" << fair_rate 
                  << " bytes/ns (" << (fair_rate * 8.0) << " Gbps)" << std::endl;
    }
    
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
        double remaining_size = chunk->get_remaining_size(); // Use remaining size as SimAI breaks into small flows
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
    
    // Set initial transmission start time
    chunk->set_transmission_start_time(event_queue->get_current_time());
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

    // Cancel all events (like old working version)
    topology->cancel_all_events();

    // Remove completed flow
    topology->remove_chunk_from_links(chunk);
    topology->active_chunks.remove(chunk);

    // Update link states and reschedule (like old working version)
    topology->update_link_states();
    topology->reschedule_active_chunks();

    // Notify completion to ASTRA-Sim
    chunk->invoke_callback();
}

void Topology::cancel_all_events() noexcept {
    const auto current_time = event_queue->get_current_time();
    for (Chunk* chunk : active_chunks) {
        double elapsed_time = current_time - chunk->get_transmission_start_time();
        double transmitted_size = elapsed_time * chunk->get_rate();
        chunk->update_remaining_size(transmitted_size);
        
        // Only cancel valid event IDs (non-zero)
        int event_id = chunk->get_completion_event_id();
        if (event_id > 0) {
            event_queue->cancel_event(event_id);
        }
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
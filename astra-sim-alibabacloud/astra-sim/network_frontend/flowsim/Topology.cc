#include "Topology.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <set>
#include <cmath> // Required for std::isfinite
#include <algorithm> // Required for std::find
#include <iomanip> // Required for std::fixed and std::setprecision

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
    if (active_links.empty()) return;

    std::set<Chunk*> fixed_chunks;

    // Progressive filling: iteratively fix flows on the current bottleneck link
    while (fixed_chunks.size() < active_chunks.size()) {
        double bottleneck_rate = std::numeric_limits<double>::max();
        std::pair<DeviceId, DeviceId> bottleneck_link;

        // 1) Locate link whose fair share is the minimum among all links
        // DEBUG: Track bottleneck identification
        static int bottleneck_debug_count = 0;
        if (bottleneck_debug_count < 5) {
            std::cerr << "[BOTTLENECK SEARCH] Round " << bottleneck_debug_count+1 << ":" << std::endl;
        }
        
        for (const auto& link_key : active_links) {
            double fair_rate = calculate_bottleneck_rate(link_key, fixed_chunks);
            if (bottleneck_debug_count < 5) {
                std::cerr << "  Link " << link_key.first << "->" << link_key.second 
                          << " rate=" << fair_rate << (fair_rate < bottleneck_rate ? " <- NEW BOTTLENECK" : "") << std::endl;
            }
            if (fair_rate < bottleneck_rate) {
                bottleneck_rate = fair_rate;
                bottleneck_link = link_key;
            }
        }
        if (bottleneck_debug_count < 5) {
            std::cerr << "  FINAL BOTTLENECK: " << bottleneck_link.first << "->" << bottleneck_link.second 
                      << " rate=" << bottleneck_rate << std::endl;
            bottleneck_debug_count++;
        }

        // No progress – should not happen, but guard against division issues
        if (bottleneck_rate == std::numeric_limits<double>::max() || bottleneck_rate <= 0 || !std::isfinite(bottleneck_rate)) {
            bottleneck_rate = 1.0;
        }

        // 2) Fix all yet-unfixed chunks on the bottleneck link to that rate
        for (Chunk* chunk : link_map[bottleneck_link]->active_chunks) {
            if (fixed_chunks.insert(chunk).second) { // newly inserted
                chunk->set_rate(bottleneck_rate);
                
                // DEBUG: Show rate assignment for first few chunks
                static int rate_debug_count = 0;
                if (rate_debug_count < 10) {
                    std::cerr << "[RATE ASSIGNMENT] Chunk " << rate_debug_count+1 
                              << " on bottleneck link " << bottleneck_link.first << "->" << bottleneck_link.second
                              << " assigned rate=" << bottleneck_rate << " bytes/ns" << std::endl;
                    rate_debug_count++;
                }
            }
        }
    }
}

// Helper: fair share a link would provide to each still-unfixed chunk
double Topology::calculate_bottleneck_rate(const std::pair<DeviceId, DeviceId>& link,
                                           const std::set<Chunk*>& fixed_chunks) {
    double remaining_bandwidth = link_map[link]->get_bandwidth();
    int active_chunks = 0;

    for (Chunk* chunk : link_map[link]->active_chunks) {
        if (fixed_chunks.find(chunk) == fixed_chunks.end()) {
            ++active_chunks;
        } else {
            // This flow's rate is already fixed; subtract its share.
            remaining_bandwidth -= chunk->get_rate();
        }
    }

    // DEBUG: Reduced rate calculation output
    static int debug_count = 0;
    
    // DEBUG: Show key bottleneck updates only when active chunks change significantly
    static int last_active_chunks = -1;
    if (link.first == 144 && link.second == 152 && active_chunks != last_active_chunks && active_chunks > 0) {
        std::cerr << "[BOTTLENECK] Link 144->152: " << active_chunks << " flows, rate=" 
                  << std::fixed << std::setprecision(3) 
                  << (active_chunks > 0 ? remaining_bandwidth / active_chunks : 0) << " Gbps" << std::endl;
        last_active_chunks = active_chunks;
    }

    return active_chunks > 0 ? remaining_bandwidth / active_chunks : std::numeric_limits<double>::max();
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
    
    // DEBUG: Track additions to bottleneck link
    static int chunks_added_to_144_152 = 0;
    
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
            // DEBUG: Track bottleneck link before/after addition
            if (src_device == 144 && dest_device == 152) {
                int before_count = link_map[link_key]->active_chunks.size();
                link_map[link_key]->active_chunks.push_back(chunk);
                int after_count = link_map[link_key]->active_chunks.size();
                
                chunks_added_to_144_152++;
                if (chunks_added_to_144_152 <= 10) {
                    std::cerr << "[ADD #" << chunks_added_to_144_152 
                              << "] Link 144->152: " << before_count << " -> " << after_count 
                              << " flows (added: " << (after_count - before_count) << ")" << std::endl;
                }
            } else {
                link_map[link_key]->active_chunks.push_back(chunk);
            }
            
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
    
    // NOTE: Rate updates happen at batch level in post_batch_completion_callback(),
    // not per individual completion for efficiency with massive NCCL flows
}



// This callback is invoked once after all chunk completions that occurred at the
// same simulated time. It updates remaining sizes, recalculates rates, and
// reschedules completion events for the still-active chunks in a single pass.
void Topology::post_batch_completion_callback(void* arg) noexcept {
    Topology* topology = static_cast<Topology*>(arg);
    topology->recalc_event_scheduled_ = false;

    const auto current_time = topology->event_queue->get_current_time();
    
    // BANDWIDTH LEAK DEBUG: Track rate recalculation after completions
    static int recalc_count = 0;
    if (++recalc_count <= 10) {
        std::cerr << "[RECALC #" << recalc_count << "] Rate recalculation triggered at " 
                  << current_time << "ns, active chunks: " << topology->active_chunks.size() 
                  << ", flag reset to FALSE" << std::endl;
    }

    // Update remaining sizes for all active chunks and cancel existing events
    for (Chunk* active_chunk : topology->active_chunks) {
        double elapsed_time = current_time - active_chunk->get_transmission_start_time();
        double transmitted_size = elapsed_time * active_chunk->get_rate();
        active_chunk->update_remaining_size(transmitted_size);
        active_chunk->set_transmission_start_time(current_time);

        if (active_chunk->get_completion_event_id() > 0) {
            topology->event_queue->cancel_event(active_chunk->get_completion_event_id());
            active_chunk->set_completion_event_id(0);
        }
    }

    // Recalculate link states only once for all active flows
    topology->update_link_states();

    // Reschedule completion events for the remaining chunks
    for (Chunk* active_chunk : topology->active_chunks) {
        double remaining_size = active_chunk->get_remaining_size();
        double new_rate = active_chunk->get_rate();
        if (new_rate <= 0) new_rate = 1.0;
        double transmission_time = remaining_size / new_rate;
        double path_latency = topology->calculate_path_latency(active_chunk);
        uint64_t completion_time = current_time + std::max(1.0, transmission_time + path_latency);
        
        // DEBUG: Show completion calculation for first few chunks (reduced)
        static int completion_debug_count = 0;
        if (completion_debug_count < 3) {
            std::cerr << "[CHUNK #" << (completion_debug_count+1) 
                      << "] size=" << (int)(remaining_size/1024) << "KB, rate=" 
                      << std::fixed << std::setprecision(1) << new_rate << "Gbps, time=" 
                      << (completion_time/1000000.0) << "ms" << std::endl;
            completion_debug_count++;
        }
        auto* chunk_ptr = static_cast<void*>(active_chunk);
        int event_id = topology->event_queue->schedule_event(completion_time, chunk_completion_callback, chunk_ptr);
        active_chunk->set_completion_event_id(event_id);
    }
}

// Batching Implementation (Performance Optimization)

void Topology::process_batch_of_chunks() {
    if (pending_chunks_.empty()) {
        return;
    }
    
    // Reset batching state
    batch_timeout_event_id_ = 0;
    last_batch_time_ = 0;
    
    // Process all pending chunks as a batch
    const auto current_time = event_queue->get_current_time();
    
    // Process all pending chunks in this temporal batch
    // DEBUG: Show temporal batch processing info (minimal)
    static int batch_debug_count = 0;
    if (++batch_debug_count <= 3) {
        std::cerr << "[BATCH #" << batch_debug_count << "] " << pending_chunks_.size() 
                  << " flows (active: " << active_chunks.size() << ")" << std::endl;
    }
    
    for (Chunk* chunk : pending_chunks_) {
        // Set topology reference and start time
        chunk->set_topology(this);
        chunk->set_transmission_start_time(current_time);
        
        // Add to active chunks list
        active_chunks.push_back(chunk);
        
        // Associate with links
        associate_chunk_with_links(chunk);
    }
    
    // Calculate rates for all chunks
    update_link_states();
    
    // Store chunks for completion event scheduling before clearing pending list
    std::vector<Chunk*> newly_added_chunks = pending_chunks_;
    
    // Clear pending chunks since they're now active
    pending_chunks_.clear();
    
    // Schedule completion events for all newly processed chunks
    for (Chunk* chunk : newly_added_chunks) {
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

// RESTORED: Working temporal batching that gave correct ~3μs FCT results
void Topology::send_with_batching(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);
    
    const auto current_time = event_queue->get_current_time();
    
    // Store chunk ownership
    active_chunks_ptrs.push_back(std::move(chunk));
    Chunk* chunk_ptr = active_chunks_ptrs.back().get();
    
    // Add to pending batch (temporal batching)
    pending_chunks_.push_back(chunk_ptr);

    // If no batch processing event is scheduled, start a new batching window
    if (batch_timeout_event_id_ == 0) {
        last_batch_time_ = current_time;
        uint64_t schedule_time = current_time + BATCH_TIMEOUT_NS;
        batch_timeout_event_id_ = event_queue->schedule_event(schedule_time, batch_timeout_callback, this);
    }
}

// RESTORED: Original working chunk completion callback
void Topology::chunk_completion_callback(void* arg) noexcept {
    Chunk* chunk = static_cast<Chunk*>(arg);
    Topology* topology = chunk->get_topology();

    // Call ASTRA-Sim callback FIRST (this is critical!)
    chunk->invoke_callback();

    // Remove this chunk from the simulation
    topology->remove_chunk_from_links(chunk);
    topology->active_chunks.remove(chunk);  // Use .remove() not .erase() for std::list

    // Schedule a single post-batch completion handler to process all remaining
    // active flows only once for this event_time.
    if (!topology->recalc_event_scheduled_ && !topology->active_chunks.empty()) {
        topology->recalc_event_scheduled_ = true;
        auto* topo_ptr = static_cast<void*>(topology);
        uint64_t now = topology->event_queue->get_current_time();
        
        // BANDWIDTH LEAK DEBUG: Track why rate recalculations stop happening
        static int recalc_schedule_count = 0;
        if (++recalc_schedule_count <= 10) {
            std::cerr << "[RECALC SCHEDULE #" << recalc_schedule_count << "] Scheduling rate recalc at " 
                      << now << "ns, active chunks: " << topology->active_chunks.size() << std::endl;
        }
        
        topology->event_queue->schedule_event(now, post_batch_completion_callback, topo_ptr);
    } else if (topology->recalc_event_scheduled_) {
        // DEBUG: Track when recalculations are skipped due to flag
        static int skip_count = 0;
        if (++skip_count <= 10) {
            std::cerr << "[RECALC SKIP #" << skip_count << "] Rate recalc already scheduled, active chunks: " 
                      << topology->active_chunks.size() << std::endl;
        }
    }
}


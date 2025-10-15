#!/usr/bin/env python3
"""
SimAI simulation runner for single topology or parallel batch execution.

Runs FlowSim, NS-3, and M4 simulations on a specified topology file.
Supports parallel execution of all workload-topology combinations.
"""

import subprocess
import argparse
import os
import sys
import time
import json
import multiprocessing as mp
from pathlib import Path
from datetime import datetime


class SimulationRunner:
    def __init__(self, simulator, topo_file, workload_file, config_file, results_dir, 
                 num_gpus=None):
        self.simulator = simulator
        self.topo_file = Path(topo_file)
        self.workload_file = workload_file
        self.config_file = config_file
        self.results_dir = Path(results_dir)
        self.num_gpus = num_gpus
        
        # Validate simulator
        valid_simulators = ['flowsim', 'ns3', 'm4']
        if simulator not in valid_simulators:
            raise ValueError(f"Simulator must be one of {valid_simulators}")
        
        # Validate topology file exists
        if not self.topo_file.exists():
            raise FileNotFoundError(f"Topology file not found: {self.topo_file}")
        
        print(f"🚀 Initializing {simulator.upper()} simulation")
        print(f"   Topology: {self.topo_file}")
        print(f"   Results: {self.results_dir}")
    
    def build_command(self, gpu_id=None):
        """Build the simulation command."""
        # Base command components
        env_vars = "AS_SEND_LAT=3 AS_NVLS_ENABLE=1"
        
        # Add GPU assignment for M4
        if self.simulator == 'm4' and gpu_id is not None:
            env_vars = f"CUDA_VISIBLE_DEVICES={gpu_id} {env_vars}"
        
        # Create output directory for this simulator
        output_subdir = self.results_dir / self.simulator
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        if self.simulator == 'flowsim':
            cmd = (f"{env_vars} ./bin/SimAI_flowsim "
                   f"-w {self.workload_file} "
                   f"-n {self.topo_file} "
                   f"-o {output_subdir}/")
        
        elif self.simulator == 'ns3':
            cmd = (f"{env_vars} ./bin/SimAI_simulator "
                   f"-t 8 "
                   f"-w {self.workload_file} "
                   f"-n {self.topo_file} "
                   f"-c {self.config_file} "
                   f"-o {output_subdir}/ "
                   f"-r")
        
        elif self.simulator == 'm4':
            cmd = (f"{env_vars} ./bin/SimAI_m4 "
                   f"-w {self.workload_file} "
                   f"-n {self.topo_file} "
                   f"-o {output_subdir}/")
        
        return cmd, output_subdir
    
    def run_simulation(self, gpu_id=None):
        """Run the simulation."""
        cmd, output_subdir = self.build_command(gpu_id)
        
        # Log file (save in the subdirectory)
        log_file = output_subdir / "run.log"
        runtime_file = output_subdir / "runtime.txt"
        
        start_time = time.time()
        
        print(f"🔄 Running {self.simulator.upper()} simulation...")
        print(f"   Command: {cmd}")
        print()
        
        try:
            # Run the command
            with open(log_file, 'w') as log:
                log.write(f"Command: {cmd}\n")
                log.write(f"Start time: {datetime.now().isoformat()}\n")
                log.write(f"{'=' * 80}\n\n")
                log.flush()
                
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,  # 1 hour timeout
                    cwd=Path.cwd()  # Run from SimAI directory
                )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                status = 'success'
                print(f"✅ Simulation completed successfully in {duration:.1f}s")
            else:
                status = 'failed'
                print(f"❌ Simulation failed (exit code {result.returncode})")
            
            # Save runtime information
            with open(runtime_file, 'w') as f:
                f.write(f"simulator: {self.simulator}\n")
                f.write(f"topology: {self.topo_file}\n")
                f.write(f"status: {status}\n")
                f.write(f"duration_seconds: {duration:.6f}\n")
                f.write(f"start_time: {datetime.fromtimestamp(start_time).isoformat()}\n")
                f.write(f"end_time: {datetime.now().isoformat()}\n")
            
            return {
                'status': status,
                'duration': duration,
                'output_dir': output_subdir,
                'log_file': log_file,
                'runtime_file': runtime_file
            }
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏱️  Simulation timed out after {duration:.1f}s")
            
            # Save runtime information for timeout
            with open(runtime_file, 'w') as f:
                f.write(f"simulator: {self.simulator}\n")
                f.write(f"topology: {self.topo_file}\n")
                f.write(f"status: timeout\n")
                f.write(f"duration_seconds: {duration:.6f}\n")
                f.write(f"start_time: {datetime.fromtimestamp(start_time).isoformat()}\n")
                f.write(f"end_time: {datetime.now().isoformat()}\n")
            
            return {
                'status': 'timeout',
                'duration': duration,
                'output_dir': output_subdir,
                'log_file': log_file,
                'runtime_file': runtime_file
            }
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Simulation error: {e}")
            
            # Save runtime information for error
            with open(runtime_file, 'w') as f:
                f.write(f"simulator: {self.simulator}\n")
                f.write(f"topology: {self.topo_file}\n")
                f.write(f"status: error\n")
                f.write(f"duration_seconds: {duration:.6f}\n")
                f.write(f"error: {str(e)}\n")
                f.write(f"start_time: {datetime.fromtimestamp(start_time).isoformat()}\n")
                f.write(f"end_time: {datetime.now().isoformat()}\n")
            
            return {
                'status': 'error',
                'duration': duration,
                'error': str(e),
                'output_dir': output_subdir,
                'log_file': log_file,
                'runtime_file': runtime_file
            }
    
    def save_summary(self, result):
        """Save summary to file."""
        summary_file = self.results_dir / f"{self.simulator}_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"SimAI Simulation Summary - {self.simulator.upper()}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Topology: {self.topo_file}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Duration: {result['duration']:.1f}s\n")
            f.write(f"Output directory: {result['output_dir']}\n")
            f.write(f"Log file: {result['log_file']}\n")
        
        print(f"📄 Summary saved to: {summary_file}")
        print(f"📁 Results directory: {result['output_dir']}")


class ParallelBatchRunner:
    """Run multiple simulations in parallel."""
    
    def __init__(self, simulator, results_base_dir, max_workers=4, config_file='./example/SimAI.conf'):
        self.simulator = simulator
        self.results_base_dir = Path(results_base_dir)
        self.max_workers = max_workers
        self.config_file = config_file
        
        # Define all workloads and topologies
        self.workloads = [
            ('W1', 'example/W1_PipelineDP_PP8_DP8.txt', 64),
            ('W2', 'example/W2_ConcurrentDP_8Rings.txt', 64),
            ('W3', 'example/W3_MoE_Imbalanced.txt', 64),
            ('W4', 'example/W4_Bursty_GA2.txt', 64),
            ('W5', 'example/W5_FullHierarchical_DP16_TP2_PP2.txt', 64),
        ]
        
        self.topologies = [
            ('T1', 'example/T1_DCN+SingleToR_64g_8gps_100Gbps_A100', 64),
            ('T2', 'example/T2_AlibabaHPN_64g_8gps_DualToR_SinglePlane_200Gbps_A100', 64),
            ('T3', 'example/T3_Spectrum-X_64g_8gps_100Gbps_A100', 64),
            ('T4', 'example/T4_Spectrum-X_64g_8gps_200Gbps_A100', 64),
            ('T5', 'example/T5_Spectrum-X_64g_8gps_50Gbps_A100', 64),
        ]
    
    def get_combinations(self):
        """Get all compatible workload-topology combinations.
        
        Returns:
            List of (workload_info, topo_info) tuples
        """
        combinations = []
        skipped = []
        
        for workload in self.workloads:
            w_name, w_file, w_gpus = workload
            for topo in self.topologies:
                t_name, t_file, t_gpus = topo
                
                # Only include matching GPU counts
                if w_gpus != t_gpus:
                    skipped.append((w_name, t_name, f"GPU mismatch: {w_gpus} vs {t_gpus}"))
                    continue
                
                combinations.append((workload, topo))
        
        if skipped:
            print(f"⚠️  Warning: {len(skipped)} incompatible combinations found (GPU count mismatch)")
            for w, t, reason in skipped:
                print(f"   {w} × {t}: {reason}")
            print()
        
        return combinations, skipped
    
    def run_single_job(self, job_info):
        """Run a single simulation job."""
        job_id, (w_name, w_file, w_gpus), (t_name, t_file, t_gpus) = job_info
        
        results_dir = self.results_base_dir / f"{w_name}_{t_name}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{job_id:2d}] Starting: {w_name} × {t_name}")
        
        start_time = time.time()
        
        try:
            runner = SimulationRunner(
                simulator=self.simulator,
                topo_file=t_file,
                workload_file=w_file,
                config_file=self.config_file,
                results_dir=results_dir
            )
            
            result = runner.run_simulation()
            duration = time.time() - start_time
            
            status_icon = '✅' if result['status'] == 'success' else '❌'
            print(f"[{job_id:2d}] {status_icon} {w_name} × {t_name} ({duration:.1f}s)")
            
            return {
                'job_id': job_id,
                'workload': w_name,
                'topology': t_name,
                'status': result['status'],
                'duration': duration,
                'results_dir': str(results_dir)
            }
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"[{job_id:2d}] ❌ {w_name} × {t_name} (error: {e})")
            
            return {
                'job_id': job_id,
                'workload': w_name,
                'topology': t_name,
                'status': 'error',
                'duration': duration,
                'error': str(e),
                'results_dir': str(results_dir)
            }
    
    def run_all(self):
        """Run all compatible simulations in parallel."""
        combinations, skipped = self.get_combinations()
        
        total_jobs = len(combinations)
        print(f"🚀 Running {total_jobs} simulations with {self.max_workers} parallel workers")
        print(f"   Simulator: {self.simulator.upper()}")
        print(f"   Results: {self.results_base_dir}")
        print()
        
        jobs = [(i+1, w, t) for i, (w, t) in enumerate(combinations)]
        
        start_time = time.time()
        
        with mp.Pool(processes=self.max_workers) as pool:
            results = pool.map(self.run_single_job, jobs)
        
        total_duration = time.time() - start_time
        
        # Save summary
        self.save_summary(results, total_duration, skipped)
        
        return results
    
    def save_summary(self, results, total_duration, skipped):
        """Save summary of all results."""
        summary_file = self.results_base_dir / f"batch_summary_{self.simulator}.json"
        
        success = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] in ['failed', 'error', 'timeout'])
        avg_duration = sum(r['duration'] for r in results) / len(results) if results else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'simulator': self.simulator,
            'total_jobs': len(results),
            'success': success,
            'failed': failed,
            'skipped': len(skipped),
            'total_duration_seconds': total_duration,
            'avg_job_duration_seconds': avg_duration,
            'max_workers': self.max_workers,
            'results': results,
            'skipped_combinations': [{'workload': w, 'topology': t, 'reason': r} 
                                     for w, t, r in skipped]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print()
        print("=" * 80)
        print(f"📊 BATCH SUMMARY - {self.simulator.upper()}")
        print("=" * 80)
        print(f"Total jobs:       {len(results)}")
        print(f"Success:          {success} ✅")
        print(f"Failed:           {failed} ❌")
        print(f"Skipped:          {len(skipped)} ⚠️")
        print(f"Total duration:   {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"Avg job duration: {avg_duration:.1f}s")
        print(f"Speedup:          {avg_duration * len(results) / total_duration:.1f}×")
        print(f"Results:          {self.results_base_dir}")
        print(f"Summary:          {summary_file}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run SimAI simulation for a single topology or all combinations in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation
  python run_simai.py flowsim --topo example/T1_Spectrum-X_64g_8gps_100Gbps_A100 --workload example/W5_FullHierarchical_DP16_TP2_PP2.txt

  # Run all 25 combinations in parallel with 4 workers
  python run_simai.py flowsim --batch --workers 4

  # NS-3 batch with 8 workers
  python run_simai.py ns3 --batch --workers 8
        """
    )
    
    parser.add_argument('simulator', choices=['flowsim', 'ns3', 'm4'],
                       help='Simulator to run')
    
    # Single simulation mode
    parser.add_argument('--topo', default='./example/topo.txt',
                       help='Topology file (default: ./example/topo.txt)')
    parser.add_argument('--workload', default='./example/microAllReduce.txt',
                       help='Workload file (default: ./example/microAllReduce.txt)')
    parser.add_argument('--config', default='./example/SimAI.conf',
                       help='Config file for NS-3 (default: ./example/SimAI.conf)')
    parser.add_argument('--results-dir', default='./results',
                       help='Output directory for results (default: ./results or ./results_batch)')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID for M4 (default: None, uses CUDA_VISIBLE_DEVICES if set)')
    
    # Batch parallel mode
    parser.add_argument('--batch', action='store_true',
                       help='Run all workload-topology combinations in parallel (25 jobs, all 64 GPUs)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers for batch mode (default: 4)')
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            # Batch parallel mode
            results_dir = args.results_dir if args.results_dir != './results' else './results_batch'
            
            batch_runner = ParallelBatchRunner(
                simulator=args.simulator,
                results_base_dir=results_dir,
                max_workers=args.workers,
                config_file=args.config
            )
            
            results = batch_runner.run_all()
            
            # Exit with success only if all jobs succeeded
            success_count = sum(1 for r in results if r['status'] == 'success')
            if success_count == len(results):
                print("\n🎉 All simulations completed successfully!")
                sys.exit(0)
            else:
                print(f"\n⚠️  {len(results) - success_count} simulation(s) failed")
                sys.exit(1)
        
        else:
            # Single simulation mode
            runner = SimulationRunner(
                simulator=args.simulator,
                topo_file=args.topo,
                workload_file=args.workload,
                config_file=args.config,
                results_dir=args.results_dir,
                num_gpus=args.gpu
            )
            
            result = runner.run_simulation(gpu_id=args.gpu)
            runner.save_summary(result)
            
            # Exit with appropriate code
            if result['status'] == 'success':
                sys.exit(0)
            else:
                sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


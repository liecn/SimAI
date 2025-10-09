#!/usr/bin/env python3
"""
Parallel sweep runner for gray failure topologies.

Runs FlowSim, NS-3, and M4 simulations in parallel.
Handles all gray failure scenarios: N=[8,24), R=[4,11) → 16*7=112 topologies.
"""

import subprocess
import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


class SweepRunner:
    def __init__(self, simulator, topo_dir, workload_file, config_file, results_dir, 
                 num_workers=None, num_gpus=None):
        self.simulator = simulator
        self.topo_dir = Path(topo_dir)
        self.workload_file = workload_file
        self.config_file = config_file
        self.results_dir = Path(results_dir)
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        
        # Validate simulator
        valid_simulators = ['flowsim', 'ns3', 'm4']
        if simulator not in valid_simulators:
            raise ValueError(f"Simulator must be one of {valid_simulators}")
        
        # Set default number of workers
        if self.num_workers is None:
            if simulator == 'm4':
                # Auto-detect available GPUs if not specified
                if num_gpus is None:
                    num_gpus = self._detect_num_gpus()
                    self.num_gpus = num_gpus
                # Default: use all available GPUs in parallel
                self.num_workers = num_gpus if num_gpus else 1
            else:
                # Use CPU count, but cap at reasonable limit
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                self.num_workers = min(cpu_count // 8, 32)
        
        if simulator == 'm4' and self.num_gpus:
            print(f"🚀 Initializing {simulator.upper()} sweep with {self.num_workers} workers on {self.num_gpus} GPUs")
        else:
            print(f"🚀 Initializing {simulator.upper()} sweep with {self.num_workers} workers")
    
    def _detect_num_gpus(self):
        """Auto-detect the number of available CUDA GPUs."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                num_gpus = len(result.stdout.strip().split('\n'))
                print(f"🔍 Auto-detected {num_gpus} GPU(s)")
                return num_gpus
            else:
                print("⚠️  Could not detect GPUs via nvidia-smi, defaulting to 1 GPU")
                return 1
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("⚠️  nvidia-smi not found or timed out, defaulting to 1 GPU")
            return 1
        except Exception as e:
            print(f"⚠️  Error detecting GPUs: {e}, defaulting to 1 GPU")
            return 1
    
    def find_topologies(self, n_range=None, r_range=None):
        """Find all gray failure topology files matching the criteria."""
        topo_files = []
        
        for topo_file in sorted(self.topo_dir.glob("gray_topo_N*_R*.txt")):
            # Skip metadata file
            if 'metadata' in topo_file.name.lower():
                continue
            
            # Parse N and R from filename
            name = topo_file.stem  # gray_topo_N8_R4
            parts = name.split('_')
            
            # Validate format: gray_topo_N{num}_R{num}
            if len(parts) != 4 or not parts[2].startswith('N') or not parts[3].startswith('R'):
                print(f"⚠️  Skipping invalid filename: {topo_file.name}")
                continue
            
            try:
                n = int(parts[2][1:])  # Remove 'N' prefix
                r = int(parts[3][1:])  # Remove 'R' prefix
            except ValueError:
                print(f"⚠️  Skipping invalid filename format: {topo_file.name}")
                continue
            
            # Filter by ranges if specified
            if n_range and n not in n_range:
                continue
            if r_range and r not in r_range:
                continue
            
            topo_files.append((n, r, topo_file))
        
        return topo_files
    
    def build_command(self, n, r, topo_file, gpu_id=None):
        """Build the simulation command."""
        # Create a subdirectory for this simulation
        # Directory will be named like: n_8_r_4_flowsim/
        
        # Base command components
        env_vars = f"AS_SEND_LAT=3 AS_NVLS_ENABLE=1 AS_M={r}"
        
        # Add GPU assignment for M4
        if self.simulator == 'm4' and gpu_id is not None:
            env_vars = f"CUDA_VISIBLE_DEVICES={gpu_id} {env_vars}"
        
        # Create unique output directory for this config
        output_subdir = self.results_dir / f"n_{n}_r_{r}_{self.simulator}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        if self.simulator == 'flowsim':
            cmd = (f"{env_vars} ./bin/SimAI_flowsim "
                   f"-w {self.workload_file} "
                   f"-n {topo_file} "
                   f"-o {output_subdir}/")
        
        elif self.simulator == 'ns3':
            cmd = (f"{env_vars} ./bin/SimAI_simulator "
                   f"-t 8 "
                   f"-w {self.workload_file} "
                   f"-n {topo_file} "
                   f"-c {self.config_file} "
                   f"-o {output_subdir}/ "
                   f"-r")
        
        elif self.simulator == 'm4':
            cmd = (f"{env_vars} ./bin/SimAI_m4 "
                   f"-w {self.workload_file} "
                   f"-n {topo_file} "
                   f"-o {output_subdir}/")
        
        return cmd, output_subdir
    
    def run_single_simulation(self, n, r, topo_file, gpu_id=None):
        """Run a single simulation."""
        cmd, output_subdir = self.build_command(n, r, topo_file, gpu_id)
        
        # Log file (save in the subdirectory)
        log_file = output_subdir / "run.log"
        runtime_file = output_subdir / "runtime.txt"
        
        start_time = time.time()
        
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
            else:
                status = 'failed'
            
            # Save runtime information
            with open(runtime_file, 'w') as f:
                f.write(f"simulator: {self.simulator}\n")
                f.write(f"n: {n}\n")
                f.write(f"r: {r}\n")
                f.write(f"status: {status}\n")
                f.write(f"duration_seconds: {duration:.6f}\n")
                f.write(f"start_time: {datetime.fromtimestamp(start_time).isoformat()}\n")
                f.write(f"end_time: {datetime.now().isoformat()}\n")
            
            return {
                'n': n, 'r': r,
                'status': status,
                'duration': duration,
                'output_dir': output_subdir,
                'log_file': log_file,
                'runtime_file': runtime_file
            }
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            
            # Save runtime information for timeout
            with open(runtime_file, 'w') as f:
                f.write(f"simulator: {self.simulator}\n")
                f.write(f"n: {n}\n")
                f.write(f"r: {r}\n")
                f.write(f"status: timeout\n")
                f.write(f"duration_seconds: {duration:.6f}\n")
                f.write(f"start_time: {datetime.fromtimestamp(start_time).isoformat()}\n")
                f.write(f"end_time: {datetime.now().isoformat()}\n")
            
            return {
                'n': n, 'r': r,
                'status': 'timeout',
                'duration': duration,
                'output_dir': output_subdir,
                'log_file': log_file,
                'runtime_file': runtime_file
            }
        
        except Exception as e:
            duration = time.time() - start_time
            
            # Save runtime information for error
            with open(runtime_file, 'w') as f:
                f.write(f"simulator: {self.simulator}\n")
                f.write(f"n: {n}\n")
                f.write(f"r: {r}\n")
                f.write(f"status: error\n")
                f.write(f"duration_seconds: {duration:.6f}\n")
                f.write(f"error: {str(e)}\n")
                f.write(f"start_time: {datetime.fromtimestamp(start_time).isoformat()}\n")
                f.write(f"end_time: {datetime.now().isoformat()}\n")
            
            return {
                'n': n, 'r': r,
                'status': 'error',
                'duration': duration,
                'error': str(e),
                'output_dir': output_subdir,
                'log_file': log_file,
                'runtime_file': runtime_file
            }
    
    def run_sweep(self, n_range=None, r_range=None):
        """Run the complete sweep."""
        # Find topologies
        topologies = self.find_topologies(n_range, r_range)
        
        if not topologies:
            print("❌ No topologies found matching criteria!")
            return
        
        print(f"📊 Found {len(topologies)} topology configurations")
        print(f"   N range: {min(t[0] for t in topologies)} - {max(t[0] for t in topologies)}")
        print(f"   R range: {min(t[1] for t in topologies)} - {max(t[1] for t in topologies)}")
        print(f"   Workers: {self.num_workers}")
        print()
        
        # Results tracking
        results = []
        completed = 0
        failed = 0
        total = len(topologies)
        
        start_time = time.time()
        
        # Run simulations
        if self.num_workers == 1:
            # Sequential execution (for debugging)
            print("🔄 Running sequentially...")
            for i, (n, r, topo_file) in enumerate(topologies, 1):
                print(f"[{i}/{total}] Running N={n}, R={r}...", end=' ', flush=True)
                result = self.run_single_simulation(n, r, topo_file)
                results.append(result)
                
                if result['status'] == 'success':
                    completed += 1
                    print(f"✅ ({result['duration']:.1f}s)")
                else:
                    failed += 1
                    print(f"❌ {result['status'].upper()}")
        
        else:
            # Parallel execution (for FlowSim, NS-3, and M4 with multiple GPUs)
            print(f"⚡ Running in parallel ({self.num_workers} workers)...")
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all jobs
                if self.simulator == 'm4' and self.num_gpus:
                    # For M4, assign each job to a GPU in round-robin fashion
                    future_to_topo = {}
                    for idx, (n, r, topo_file) in enumerate(topologies):
                        gpu_id = idx % self.num_gpus
                        future = executor.submit(self.run_single_simulation, n, r, topo_file, gpu_id)
                        future_to_topo[future] = (n, r, gpu_id)
                else:
                    # For FlowSim and NS-3, no GPU assignment needed
                    future_to_topo = {
                        executor.submit(self.run_single_simulation, n, r, topo_file): (n, r, None)
                        for n, r, topo_file in topologies
                    }
                
                # Process completed jobs
                for future in as_completed(future_to_topo):
                    n, r, gpu_id = future_to_topo[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['status'] == 'success':
                            completed += 1
                            status_icon = "✅"
                        else:
                            failed += 1
                            status_icon = "❌"
                        
                        progress = len(results)
                        gpu_info = f" [GPU {gpu_id}]" if gpu_id is not None else ""
                        print(f"{status_icon} [{progress}/{total}] N={n}, R={r}{gpu_info}: "
                              f"{result['status']} ({result['duration']:.1f}s)")
                    
                    except Exception as e:
                        failed += 1
                        print(f"❌ [{len(results)+1}/{total}] N={n}, R={r}: ERROR - {e}")
        
        elapsed = time.time() - start_time
        
        # Summary
        print()
        print("=" * 80)
        print(f"📊 SWEEP SUMMARY - {self.simulator.upper()}")
        print("=" * 80)
        print(f"Total configurations: {total}")
        print(f"✅ Completed: {completed}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        if completed > 0:
            print(f"⏱️  Average time per simulation: {elapsed/completed:.1f}s")
        print()
        
        # Save results summary
        self.save_summary(results, elapsed)
        
        # Show failures
        if failed > 0:
            print("❌ Failed simulations:")
            for result in results:
                if result['status'] != 'success':
                    print(f"   N={result['n']}, R={result['r']}: "
                          f"{result['status']} - {result.get('log_file', 'N/A')}")
        
        return results
    
    def save_summary(self, results, elapsed):
        """Save summary to file."""
        summary_file = self.results_dir / f"{self.simulator}_sweep_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Gray Failure Sweep Summary - {self.simulator.upper()}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)\n")
            f.write(f"\n")
            f.write(f"{'N':>3} {'R':>3} {'Status':<10} {'Duration(s)':>12} {'Output Dir':<60}\n")
            f.write(f"{'-' * 80}\n")
            
            for result in sorted(results, key=lambda x: (x['n'], x['r'])):
                f.write(f"{result['n']:>3} {result['r']:>3} "
                       f"{result['status']:<10} "
                       f"{result['duration']:>12.1f} "
                       f"{result.get('output_dir', 'N/A')}\n")
        
        print(f"📄 Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run gray failure sweep for SimAI simulators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single scenario: N=10, R=5 with FlowSim
  ./run_gray_failure_sweep.py flowsim --n 10 --r 5

  # Run a single scenario: N=12, R=8 with NS-3
  ./run_gray_failure_sweep.py ns3 --n 12 --r 8

  # Run a single scenario: N=15, R=6 with M4
  ./run_gray_failure_sweep.py m4 --n 15 --r 6

  # Run FlowSim on all topologies (parallel)
  ./run_gray_failure_sweep.py flowsim

  # Run NS-3 with 8 parallel workers
  ./run_gray_failure_sweep.py ns3 --workers 8

  # Run M4 on 4 GPUs in parallel (default)
  ./run_gray_failure_sweep.py m4

  # Run M4 on 2 GPUs in parallel
  ./run_gray_failure_sweep.py m4 --num-gpus 2

  # Test subset: N=[8,12), R=[4,8)
  ./run_gray_failure_sweep.py flowsim --n-min 8 --n-max 12 --r-min 4 --r-max 8
        """
    )
    
    parser.add_argument('simulator', choices=['flowsim', 'ns3', 'm4'],
                       help='Simulator to run')
    parser.add_argument('--topo-dir', default='./example/gray_failures',
                       help='Directory containing topology files')
    parser.add_argument('--workload', default='./example/microAllReduce.txt',
                       help='Workload file')
    parser.add_argument('--config', default='./example/SimAI.conf',
                       help='Config file (for NS-3)')
    parser.add_argument('--results-dir', default='./results_gray_failures',
                       help='Output directory for results')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    parser.add_argument('--num-gpus', type=int, default=None,
                       help='Number of GPUs for M4 (default: auto-detect via nvidia-smi)')
    parser.add_argument('--n', type=int, default=None,
                       help='Run single scenario with specific N value (overrides --n-min/--n-max)')
    parser.add_argument('--r', type=int, default=None,
                       help='Run single scenario with specific R value (overrides --r-min/--r-max)')
    parser.add_argument('--n-min', type=int, default=2,
                       help='Minimum N value (number of degraded GPUs)')
    parser.add_argument('--n-max', type=int, default=17,
                       help='Maximum N value (exclusive)')
    parser.add_argument('--r-min', type=int, default=4,
                       help='Minimum R value (reduction factor)')
    parser.add_argument('--r-max', type=int, default=11,
                       help='Maximum R value (exclusive)')
    
    args = parser.parse_args()
    
    # Build ranges or single scenario
    if args.n is not None and args.r is not None:
        # Single scenario mode
        n_range = [args.n]
        r_range = [args.r]
        print(f"🎯 Running single scenario: N={args.n}, R={args.r}")
    elif args.n is not None or args.r is not None:
        # Error: both must be specified for single scenario
        print("❌ Error: Both --n and --r must be specified to run a single scenario")
        sys.exit(1)
    else:
        # Range mode (default)
        n_range = range(args.n_min, args.n_max) if args.n_min or args.n_max else None
        r_range = range(args.r_min, args.r_max) if args.r_min or args.r_max else None
    
    # GPU count for M4 (will be auto-detected if None)
    num_gpus = args.num_gpus
    
    # Create runner
    runner = SweepRunner(
        simulator=args.simulator,
        topo_dir=args.topo_dir,
        workload_file=args.workload,
        config_file=args.config,
        results_dir=args.results_dir,
        num_workers=args.workers,
        num_gpus=num_gpus
    )
    
    # Run sweep
    try:
        runner.run_sweep(n_range=n_range, r_range=r_range)
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


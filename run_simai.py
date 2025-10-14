#!/usr/bin/env python3
"""
SimAI simulation runner for single topology.

Runs FlowSim, NS-3, and M4 simulations on a specified topology file.
"""

import subprocess
import argparse
import os
import sys
import time
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


def main():
    parser = argparse.ArgumentParser(
        description='Run SimAI simulation for a single topology',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run FlowSim on topo.txt
  python run_simai.py flowsim --topo example/topo.txt

  # Run NS-3 on topo.txt
  python run_simai.py ns3 --topo example/topo.txt

  # Run M4 on topo.txt with GPU 0
  python run_simai.py m4 --topo example/topo.txt --gpu 0
        """
    )
    
    parser.add_argument('simulator', choices=['flowsim', 'ns3', 'm4'],
                       help='Simulator to run')
    parser.add_argument('--topo', default='./example/topo.txt',
                       help='Topology file (default: ./example/topo.txt)')
    parser.add_argument('--workload', default='./example/microAllReduce.txt',
                       help='Workload file (default: ./example/microAllReduce.txt)')
    parser.add_argument('--config', default='./example/SimAI.conf',
                       help='Config file for NS-3 (default: ./example/SimAI.conf)')
    parser.add_argument('--results-dir', default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID for M4 (default: None, uses CUDA_VISIBLE_DEVICES if set)')
    
    args = parser.parse_args()
    
    # Create runner
    runner = SimulationRunner(
        simulator=args.simulator,
        topo_file=args.topo,
        workload_file=args.workload,
        config_file=args.config,
        results_dir=args.results_dir,
        num_gpus=args.gpu
    )
    
    # Run simulation
    try:
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


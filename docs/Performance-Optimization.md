# Performance Optimization

Comprehensive guide to optimizing Open-Sourcefy Matrix pipeline performance for faster execution and better resource utilization.

## Performance Overview

### Current Performance Baseline

- **Total Pipeline Time**: 15-30 minutes for typical 5MB binary
- **Memory Usage**: 4-8GB peak (16GB+ recommended)
- **CPU Utilization**: 80-95% during parallel agent execution
- **Disk I/O**: 2-5GB temporary files during processing
- **Success Rate**: 100% (16/16 agents operational)

### Performance Targets

- **Execution Time**: <15 minutes for typical binary
- **Memory Efficiency**: <6GB peak usage
- **CPU Optimization**: >95% utilization during parallel phases
- **I/O Optimization**: <3GB temporary storage
- **Quality Maintenance**: >85% reconstruction accuracy

## System Optimization

### Hardware Recommendations

#### Optimal Configuration
```
CPU: Intel i7/i9 or AMD Ryzen 7/9 (8+ cores)
RAM: 32GB DDR4-3200 or faster
Storage: NVMe SSD (1TB+) for temporary files
GPU: Not required (CPU-intensive workload)
Network: Stable connection for AI integration
```

#### Minimum Configuration
```
CPU: Intel i5 or AMD Ryzen 5 (4+ cores)
RAM: 16GB DDR4-2400
Storage: SATA SSD (500GB+)
```

### Operating System Tuning

#### Windows Performance Settings
```powershell
# Enable high performance power plan
powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Increase virtual memory
# System Properties → Advanced → Performance Settings → Advanced → Virtual Memory
# Set to 32GB fixed size on SSD

# Disable unnecessary services
sc config "Windows Search" start= disabled
sc config "Superfetch" start= disabled
```

#### Linux Performance Tuning
```bash
# CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize I/O scheduler
echo deadline | sudo tee /sys/block/sda/queue/scheduler

# Increase VM parameters
echo "vm.swappiness=10" >> /etc/sysctl.conf
echo "vm.dirty_ratio=15" >> /etc/sysctl.conf
```

## Pipeline Configuration Optimization

### High-Performance Configuration

#### Speed-Optimized Config (`config-performance.yaml`)
```yaml
application:
  debug_mode: false
  log_level: "WARNING"  # Reduced logging overhead

agents:
  timeout: 200  # Reduced timeouts
  retry_count: 1  # Fewer retries
  parallel_execution: true
  max_parallel_agents: 8  # Increase based on CPU cores
  quality_threshold: 0.70  # Slightly relaxed for speed
  fail_fast: true

pipeline:
  execution_mode: "performance"
  validation_level: "standard"  # Reduced validation
  cache_results: true  # Enable aggressive caching
  cleanup_temp_files: false  # Defer cleanup

ghidra:
  headless_timeout: 300  # Reduced Ghidra timeout
  java_heap_size: "8G"  # Increase for faster processing
  analysis_timeout: 180

performance:
  enable_jit_compilation: true
  preload_libraries: true
  optimize_memory_usage: true
  
  cache:
    enable_result_cache: true
    cache_directory: "/tmp/openSourcefy_cache"  # RAM disk if available
    max_cache_size: "10G"
    
  parallel:
    max_agents_parallel: 8
    enable_numa_awareness: true
    cpu_affinity: "auto"
    
  io:
    use_async_io: true
    buffer_size: 1048576  # 1MB buffers
    prefetch_enabled: true

logging:
  level: "WARNING"
  destinations:
    console: false  # Disable console output
    file: true
  file:
    path: "/tmp/openSourcefy_performance.log"
```

#### Memory-Optimized Config (`config-memory.yaml`)
```yaml
agents:
  max_parallel_agents: 2  # Reduce parallelism
  
pipeline:
  cache_results: false  # Disable caching to save memory

ghidra:
  java_heap_size: "2G"  # Reduced heap size

performance:
  optimize_memory_usage: true
  
  cache:
    enable_result_cache: false
    max_cache_size: "1G"
    
  parallel:
    max_agents_parallel: 2
    
memory_management:
  gc_frequency: "aggressive"
  memory_limit: "8G"
  swap_usage: "minimal"
```

### Agent-Specific Optimizations

#### Agent 1 (Sentinel) - Binary Analysis
```python
# Optimized import table processing
def optimized_import_analysis(self, binary_path: str) -> ImportAnalysis:
    """Memory-efficient import table analysis"""
    
    # Use memory mapping for large files
    with mmap.mmap(open(binary_path, 'rb').fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
        # Process in chunks to reduce memory usage
        chunk_size = 1024 * 1024  # 1MB chunks
        for offset in range(0, len(mmapped_file), chunk_size):
            chunk = mmapped_file[offset:offset + chunk_size]
            # Process chunk efficiently
```

#### Agent 5 (Neo) - Ghidra Integration
```python
# Ghidra performance optimization
def optimize_ghidra_execution(self, binary_path: str) -> None:
    """Optimize Ghidra for performance"""
    
    # Pre-allocate large heap
    java_opts = [
        "-Xmx8g",  # Maximum heap size
        "-Xms4g",  # Initial heap size
        "-XX:+UseG1GC",  # G1 garbage collector
        "-XX:MaxGCPauseMillis=200",  # Reduce GC pauses
        "-XX:+UseStringDeduplication"  # Memory optimization
    ]
    
    # Disable unnecessary analysis
    ghidra_script = """
    analyzeHeadless.bat project_dir project_name -import {} -postScript analyze_fast.java -deleteProject
    """.format(binary_path)
```

#### Agent 9 (Commander Locke) - Compilation
```python
# Parallel compilation optimization
def optimize_compilation(self, source_files: List[str]) -> CompilationResult:
    """Optimize compilation using parallel builds"""
    
    # Use all available CPU cores
    cpu_count = os.cpu_count()
    
    # MSBuild parallel compilation
    msbuild_cmd = [
        self.msbuild_path,
        "project.vcxproj",
        f"/m:{cpu_count}",  # Parallel build
        "/p:Configuration=Release",
        "/p:Platform=x64",
        "/p:PreferredToolArchitecture=x64",
        "/p:UseMultiToolTask=true",
        "/p:EnforceProcessCountAcrossBuilds=true"
    ]
```

## Caching Strategies

### Multi-Level Caching

#### Level 1: In-Memory Cache
```python
class InMemoryCache:
    """Fast in-memory caching for frequently accessed data"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key: str) -> Any:
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
```

#### Level 2: Disk Cache
```python
class DiskCache:
    """Persistent disk caching for analysis results"""
    
    def __init__(self, cache_dir: str = "/tmp/openSourcefy_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, binary_path: str, agent_id: int) -> str:
        """Generate cache key from binary hash and agent ID"""
        with open(binary_path, 'rb') as f:
            binary_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        return f"{agent_id}_{binary_hash}"
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def set(self, cache_key: str, data: Dict[str, Any]) -> None:
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
```

#### Level 3: Distributed Cache (Redis)
```python
import redis

class DistributedCache:
    """Redis-based distributed caching for team environments"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = 3600 * 24 * 7  # 1 week TTL
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        data = self.redis_client.get(f"openSourcefy:{cache_key}")
        return json.loads(data) if data else None
    
    def set(self, cache_key: str, data: Dict[str, Any]) -> None:
        serialized = json.dumps(data)
        self.redis_client.setex(f"openSourcefy:{cache_key}", self.ttl, serialized)
```

## Parallel Processing Optimization

### Agent Batching Strategy

#### Optimized Execution Batches
```python
def optimize_agent_batches(self, selected_agents: List[int]) -> List[List[int]]:
    """Optimize agent execution batches for performance"""
    
    # Dependency-aware batching with CPU utilization optimization
    optimized_batches = [
        [0],  # Master (sequential)
        [1],  # Foundation (sequential) 
        [2, 3, 4],  # Parallel foundation (3 cores)
        [5, 6, 7, 8],  # Parallel advanced (4 cores)
        [9],  # Compilation (sequential, CPU-intensive)
        [10, 11, 12, 13],  # Parallel reconstruction (4 cores)
        [14, 15, 16]  # Parallel QA (3 cores)
    ]
    
    return optimized_batches

def execute_batch_optimized(self, agent_batch: List[int], context: Dict[str, Any]) -> Dict[int, AgentResult]:
    """Execute agent batch with CPU affinity and priority optimization"""
    
    # Set high priority for pipeline process
    import psutil
    process = psutil.Process()
    process.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
    
    # Execute with optimized thread pool
    with ThreadPoolExecutor(
        max_workers=len(agent_batch),
        thread_name_prefix="matrix_agent"
    ) as executor:
        
        # Submit with CPU affinity
        futures = {}
        for i, agent_id in enumerate(agent_batch):
            # Assign CPU affinity if available
            if hasattr(os, 'sched_setaffinity'):
                cpu_id = i % os.cpu_count()
                os.sched_setaffinity(0, {cpu_id})
            
            future = executor.submit(self._execute_single_agent, agent_id, context)
            futures[agent_id] = future
        
        # Collect results
        results = {}
        for agent_id, future in futures.items():
            results[agent_id] = future.result()
        
        return results
```

### I/O Optimization

#### Asynchronous File Operations
```python
import asyncio
import aiofiles

class AsyncFileManager:
    """Asynchronous file operations for better I/O performance"""
    
    async def read_binary_chunks(self, file_path: str, chunk_size: int = 1048576) -> AsyncGenerator[bytes, None]:
        """Read binary file in chunks asynchronously"""
        async with aiofiles.open(file_path, 'rb') as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    async def write_analysis_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Write analysis results asynchronously"""
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(json.dumps(results, indent=2))
    
    async def parallel_file_processing(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files in parallel"""
        tasks = [self.process_single_file(path) for path in file_paths]
        return await asyncio.gather(*tasks)
```

## Monitoring and Profiling

### Performance Monitoring

#### Real-Time Performance Metrics
```python
class PerformanceMonitor:
    """Real-time performance monitoring for pipeline execution"""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'agent_timings': {},
            'bottlenecks': []
        }
        self.start_time = time.time()
    
    def start_agent_monitoring(self, agent_id: int) -> None:
        """Start monitoring specific agent performance"""
        self.metrics['agent_timings'][agent_id] = {
            'start_time': time.time(),
            'cpu_start': psutil.cpu_percent(),
            'memory_start': psutil.virtual_memory().percent
        }
    
    def end_agent_monitoring(self, agent_id: int) -> Dict[str, float]:
        """End monitoring and calculate agent performance metrics"""
        if agent_id not in self.metrics['agent_timings']:
            return {}
        
        start_data = self.metrics['agent_timings'][agent_id]
        end_time = time.time()
        
        return {
            'execution_time': end_time - start_data['start_time'],
            'cpu_usage': psutil.cpu_percent() - start_data['cpu_start'],
            'memory_usage': psutil.virtual_memory().percent - start_data['memory_start']
        }
    
    def identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Analyze agent timings
        for agent_id, timing in self.metrics['agent_timings'].items():
            if timing.get('execution_time', 0) > 300:  # 5 minutes
                bottlenecks.append(f"Agent {agent_id}: Long execution time")
        
        # Check system resources
        if psutil.virtual_memory().percent > 90:
            bottlenecks.append("High memory usage")
        
        if psutil.cpu_percent(interval=1) > 95:
            bottlenecks.append("High CPU usage")
        
        return bottlenecks
```

#### Performance Profiling
```python
import cProfile
import pstats
from memory_profiler import profile

class PipelineProfiler:
    """Comprehensive profiling for pipeline optimization"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        
    def profile_agent_execution(self, agent_id: int, execution_func, *args, **kwargs):
        """Profile individual agent execution"""
        self.profiler.enable()
        result = execution_func(*args, **kwargs)
        self.profiler.disable()
        
        # Generate profile report
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        profile_file = f"profiles/agent_{agent_id}_profile.txt"
        with open(profile_file, 'w') as f:
            stats.print_stats(f)
        
        return result
    
    @profile
    def memory_profile_pipeline(self, pipeline_func, *args, **kwargs):
        """Memory profiling for pipeline execution"""
        return pipeline_func(*args, **kwargs)
```

## Performance Tuning Commands

### Performance Mode Execution
```bash
# High-performance mode
export MATRIX_PERFORMANCE_MODE=high
export MATRIX_PARALLEL_AGENTS=8
export MATRIX_CACHE_ENABLED=true
python main.py --fast --max-parallel 8

# Memory-optimized mode
export MATRIX_MEMORY_LIMIT=8G
export MATRIX_PARALLEL_AGENTS=2
python main.py --optimize-memory --max-memory 8G

# I/O optimized mode
export MATRIX_TEMP_DIR=/tmp/openSourcefy  # Use RAM disk
export MATRIX_ASYNC_IO=true
python main.py --optimize-io --temp-dir /tmp/openSourcefy

# CPU optimized mode
export MATRIX_CPU_AFFINITY=true
export MATRIX_HIGH_PRIORITY=true
python main.py --optimize-cpu --benchmark

# Profiling mode
python main.py --profile --benchmark --generate-report
```

### Performance Testing
```bash
# Benchmark different configurations
python tools/benchmark.py --config config-performance.yaml
python tools/benchmark.py --config config-memory.yaml
python tools/benchmark.py --config config-balanced.yaml

# Compare performance across binaries
python tools/performance_test.py --binary-set test_suite/

# Generate performance report
python tools/generate_performance_report.py --output performance_report.html
```

## Performance Metrics and Targets

### Key Performance Indicators (KPIs)

| Metric | Current | Target | Optimized |
|--------|---------|--------|-----------|
| Total Pipeline Time | 15-30 min | <15 min | 8-12 min |
| Memory Peak Usage | 6-8 GB | <6 GB | 4-5 GB |
| CPU Utilization | 80-95% | >95% | 98%+ |
| Disk I/O | 2-5 GB | <3 GB | 1-2 GB |
| Cache Hit Rate | N/A | >80% | 90%+ |
| Agent Success Rate | 100% | 100% | 100% |

### Performance Regression Testing
```python
class PerformanceRegression:
    """Automated performance regression testing"""
    
    def __init__(self, baseline_metrics: Dict[str, float]):
        self.baseline = baseline_metrics
        self.threshold = 0.1  # 10% regression threshold
    
    def test_performance_regression(self, current_metrics: Dict[str, float]) -> bool:
        """Test for performance regressions"""
        regressions = []
        
        for metric, baseline_value in self.baseline.items():
            current_value = current_metrics.get(metric, 0)
            regression = (current_value - baseline_value) / baseline_value
            
            if regression > self.threshold:
                regressions.append(f"{metric}: {regression:.2%} regression")
        
        if regressions:
            raise PerformanceRegressionError(f"Performance regressions detected: {regressions}")
        
        return True
```

---

**Related**: [[Configuration Guide|Configuration-Guide]] - System configuration options  
**Next**: [[Troubleshooting]] - Performance issue resolution
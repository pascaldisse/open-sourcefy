# Research Unknowns and Investigation TODO

## Critical Unknowns - Need Research

### 1. **Original Binary Functionality** 🔍
**What we don't know about launcher.exe:**
- What does it actually do? (Game launcher, updater, downloader?)
- Does it have a GUI or is it console-based?
- What servers does it connect to?
- What files does it manage/download/patch?
- Does it launch other processes?
- How does it handle authentication/security?
- What configuration files does it use?
- Does it run as a service or standalone?

**Research needed:**
- Run the original launcher.exe in a VM to observe behavior
- Use Process Monitor to see file/registry/network activity
- Use network monitoring to see what connections it makes
- Check for GUI using dependency walker
- Analyze command line arguments it accepts

### 2. **Ghidra Decompilation Results** 🔍  
**What we don't know about function extraction:**
- Why aren't Agents 3 & 5 extracting functions from Ghidra?
- What does Ghidra actually decompile from this binary?
- Are there anti-analysis protections preventing decompilation?
- What's the actual C/assembly code that Ghidra generates?

**Research needed:**
- Manually run Ghidra on launcher.exe
- Check if binary is packed/obfuscated
- Review Agent 3 & 5 logs for Ghidra failures
- Test Ghidra decompilation on simpler binaries first
- Check if Java/Ghidra integration is working properly

### 3. **PE Binary Structure Deep Dive** 🔍
**What we don't know about the binary format:**
- What's actually in those 5.3MB? (resources, code, data?)
- What embedded resources exist? (images, dialogs, strings?)
- What DLLs are actually imported and what functions?
- Is there debug information embedded?
- What's the actual entry point and initialization code?
- Are there multiple code sections?

**Research needed:**
- Use PE analysis tools (PE Explorer, CFF Explorer)
- Extract all embedded resources manually
- Analyze import table in detail
- Check for debug symbols/PDB files
- Map out memory sections and their purposes

### 4. **Binary Reconstruction Approach** 🔍
**What we don't know about the best strategy:**
- Should we focus on functional reconstruction vs binary matching?
- Is 95% binary matching even realistic for any source-based approach?
- Would binary patching/modification be more effective?
- Could we inject our code into the original binary instead?
- What do other reverse engineering tools achieve?

**Research needed:**
- Study other decompilation/reconstruction tools
- Research binary diffing methodologies
- Look into code injection techniques
- Investigate hybrid approaches (patch + reconstruct)
- Check academic papers on binary reconstruction

### 5. **Resource Integration Methods** 🔍
**What we don't know about resource handling:**
- How should extracted strings/resources be integrated into C code?
- Should resources be compiled into the binary or loaded externally?
- What's the proper way to handle embedded bitmaps/dialogs?
- How do we recreate the resource section structure?

**Research needed:**
- Study Windows resource compilation (.rc files)
- Learn about resource embedding in MSVC
- Understand difference between compiled vs external resources
- Research resource extraction and recreation workflows

### 6. **Compilation Configuration** 🔍
**What we don't know about matching compilation:**
- What compiler was used for the original? (MSVC version, GCC?)
- What compilation flags should we use?
- What linking options affect binary structure?
- How do we match the original's optimization level?
- What affects PE header generation?

**Research needed:**
- Use tools to detect original compiler (DIE, PEiD)
- Research MSVC compilation flags for binary structure
- Study linker options that affect PE layout
- Test different optimization levels
- Compare generated PE headers with original

### 7. **Matrix Agent Workflow** 🔍
**What we don't know about agent coordination:**
- Why isn't function data flowing from Agent 5 to Agent 9?
- What data format should agents use to communicate?
- Are we missing prerequisite validations?
- Should agents 3 & 5 run before 8 & 9?
- What's the optimal agent execution order?

**Research needed:**
- Trace data flow between agents with debug logging
- Review agent dependency mappings
- Test different execution orders
- Validate shared memory/context structures
- Check agent result serialization

### 8. **Validation Methodology** 🔍
**What we don't know about realistic validation:**
- What percentage match is achievable with source reconstruction?
- Which validation tasks are actually important vs cosmetic?
- How do other tools measure reconstruction success?
- Should we validate functionality vs binary structure?
- What are industry standards for decompilation success?

**Research needed:**
- Study other decompilation tool validation methods
- Research binary similarity metrics used in malware analysis
- Look into functional equivalence testing
- Check academic standards for reverse engineering
- Survey commercial decompilation tool success rates

## Technical Investigation Priorities

### High Priority 🔴
1. **Run original launcher.exe in controlled environment** - Understand what it actually does
2. **Manual Ghidra analysis** - See what actual decompilation looks like
3. **PE structure analysis** - Map out the 5.3MB binary contents
4. **Agent 3 & 5 debugging** - Fix function extraction pipeline

### Medium Priority 🟡  
1. **Resource extraction improvement** - Better integration of Agent 8 results
2. **Compilation flag research** - Match original compiler characteristics
3. **Binary similarity metrics** - Research realistic success thresholds
4. **Alternative reconstruction approaches** - Binary patching vs source generation

### Low Priority 🟢
1. **Performance optimization** - Speed up pipeline execution
2. **UI development** - Better reporting and visualization
3. **Documentation completion** - Comprehensive user guides
4. **Test case expansion** - More binary types beyond launcher.exe

## Specific Research Questions

### Functional Analysis
- **Q1**: What happens when you run `launcher.exe --help` or with various arguments?
- **Q2**: What network traffic does it generate?
- **Q3**: What files does it create/modify?
- **Q4**: Does it have any GUI components?

### Technical Analysis  
- **Q5**: What does `objdump -x launcher.exe` show for imports/exports?
- **Q6**: What does `strings launcher.exe` reveal beyond what Agent 8 found?
- **Q7**: Can `radare2` or `IDA Free` provide better decompilation than Ghidra?
- **Q8**: What does `pe-tree` or similar tools show for PE structure?

### Approach Validation
- **Q9**: What percentage similarity do commercial decompilers achieve?
- **Q10**: Are there academic papers on measuring decompilation success?
- **Q11**: How do malware researchers validate binary reconstruction?
- **Q12**: What would a "good enough" reconstruction look like for practical use?

## Success Criteria for Research

### Functional Understanding
- [ ] Document what launcher.exe actually does in detail
- [ ] Identify core functionality vs auxiliary features  
- [ ] Map out file/network/registry dependencies
- [ ] Understand user interaction model

### Technical Improvement
- [ ] Get Ghidra decompilation working properly
- [ ] Extract meaningful functions for Agent 9 to use
- [ ] Improve resource integration beyond simple string lists
- [ ] Achieve >70% binary similarity with functional equivalence

### Validation Framework
- [ ] Define realistic success metrics for source-based reconstruction
- [ ] Implement functionality-based validation alongside binary matching
- [ ] Create test harness for iterative improvement
- [ ] Establish industry-standard comparison benchmarks

This research should provide the missing knowledge needed to either fix the current approach or pivot to a more effective reconstruction strategy.
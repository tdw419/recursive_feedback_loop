Perform a deep architectural audit of the project.

For each issue found:
1. Specific file, function, and line numbers
2. Why this compounds at scale
3. A concrete fix with code

Focus areas:
- Abstraction boundaries (leaky abstractions, misplaced responsibilities)
- Error handling (swallowed errors, missing error paths, unwrap bombs)
- State management (shared mutable state, race conditions, stale references)
- Performance hot paths (unnecessary allocations, O(n²) where O(n) is possible)
- Testing gaps (untested edge cases, missing integration tests)

Read the actual source files. Don't guess — cite real code with real line numbers.

Project description: {{description}}
Key source directories: {{source_dirs}}

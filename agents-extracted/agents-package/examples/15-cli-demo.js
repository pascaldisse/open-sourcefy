/**
 * Example 15: CLI Usage Demo
 * 
 * This example demonstrates various CLI commands and features
 * Run this file to see example CLI commands you can use
 */

import chalk from 'chalk';

console.log(chalk.bold('\n🚀 Multi-Agent System CLI Demo\n'));

console.log('This demo shows example CLI commands for managing the multi-agent system.\n');

console.log(chalk.cyan('Installation:'));
console.log('  npm install -g .');
console.log('  # or use directly with: node src/cli/cli.js\n');

console.log(chalk.cyan('Basic Commands:'));
console.log(chalk.gray('  # Start the system'));
console.log('  gaia-mas start');
console.log('');
console.log(chalk.gray('  # Start with dashboard'));
console.log('  gaia-mas start --dashboard --port 3001');
console.log('');
console.log(chalk.gray('  # Start with learning enabled'));
console.log('  gaia-mas start --dashboard --learning');
console.log('');
console.log(chalk.gray('  # Check system status'));
console.log('  gaia-mas status');
console.log('  gaia-mas status --agents      # Show detailed agent status');
console.log('  gaia-mas status --performance # Show performance metrics');
console.log('');
console.log(chalk.gray('  # Stop the system'));
console.log('  gaia-mas stop\n');

console.log(chalk.cyan('Task Execution:'));
console.log(chalk.gray('  # Execute a simple task'));
console.log('  gaia-mas execute "Write unit tests for user service"');
console.log('');
console.log(chalk.gray('  # Execute with options'));
console.log('  gaia-mas execute "Fix memory leak" --type bug_fixing --priority high');
console.log('');
console.log(chalk.gray('  # Execute with context'));
console.log('  gaia-mas execute "Review code" --type code_review --context \'{"file": "auth.js"}\'\n');

console.log(chalk.cyan('Agent Management:'));
console.log(chalk.gray('  # List all agents'));
console.log('  gaia-mas agent list');
console.log('');
console.log(chalk.gray('  # Get agent details'));
console.log('  gaia-mas agent details 1');
console.log('  gaia-mas agent details 0  # Master coordinator\n');

console.log(chalk.cyan('Learning System:'));
console.log(chalk.gray('  # Get learning insights'));
console.log('  gaia-mas learning insights 1');
console.log('');
console.log(chalk.gray('  # Share knowledge between agents'));
console.log('  gaia-mas learning share 1 3  # Share from Agent 1 to Agent 3\n');

console.log(chalk.cyan('Cache Management:'));
console.log(chalk.gray('  # View cache statistics'));
console.log('  gaia-mas cache stats');
console.log('');
console.log(chalk.gray('  # Clear cache'));
console.log('  gaia-mas cache clear');
console.log('  gaia-mas cache clear "task-result"  # Clear by pattern\n');

console.log(chalk.cyan('Interactive Mode:'));
console.log(chalk.gray('  # Start interactive CLI'));
console.log('  gaia-mas interactive\n');

console.log(chalk.yellow('Example Workflow:\n'));

console.log('1. Start the system with monitoring:');
console.log(chalk.green('   $ gaia-mas start --dashboard --learning\n'));

console.log('2. Execute some tasks:');
console.log(chalk.green('   $ gaia-mas execute "Create API documentation"'));
console.log(chalk.green('   $ gaia-mas execute "Write integration tests" --priority high'));
console.log(chalk.green('   $ gaia-mas execute "Optimize database queries" --type performance_optimization\n'));

console.log('3. Monitor performance:');
console.log(chalk.green('   $ gaia-mas status --performance'));
console.log(chalk.green('   $ gaia-mas cache stats\n'));

console.log('4. Check agent learning:');
console.log(chalk.green('   $ gaia-mas learning insights 1'));
console.log(chalk.green('   $ gaia-mas learning insights 9  # Performance optimizer agent\n'));

console.log('5. View detailed agent status:');
console.log(chalk.green('   $ gaia-mas agent list'));
console.log(chalk.green('   $ gaia-mas agent details 2\n'));

console.log('6. Use interactive mode for exploration:');
console.log(chalk.green('   $ gaia-mas interactive'));
console.log(chalk.gray('   > help'));
console.log(chalk.gray('   > status'));
console.log(chalk.gray('   > execute'));
console.log(chalk.gray('   > agents'));
console.log(chalk.gray('   > perf'));
console.log(chalk.gray('   > exit\n'));

console.log(chalk.bold('📝 Tips:\n'));
console.log('• Use --help with any command for more information');
console.log('• The dashboard provides real-time visualization at http://localhost:3001');
console.log('• Learning insights improve over time as agents process more tasks');
console.log('• Cache statistics help identify optimization opportunities');
console.log('• Interactive mode is great for exploration and quick tasks\n');

console.log(chalk.cyan('For more information:'));
console.log('  gaia-mas --help');
console.log('  gaia-mas <command> --help\n');
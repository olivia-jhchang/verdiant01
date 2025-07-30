# Technology Stack

## Core Technologies
- **IDE**: Visual Studio Code
- **AI Assistant**: Kiro with MCP (Model Context Protocol) support
- **Configuration**: JSON-based settings

## Development Environment
- VSCode workspace with Kiro extension
- MCP server integration enabled
- Cross-platform development support

## Configuration Files
- `.vscode/settings.json` - VSCode workspace settings
- `.kiro/steering/*.md` - AI assistant steering rules
- `.kiro/settings/mcp.json` - MCP server configurations (when needed)

## Common Commands
Since this is a flexible development workspace, specific build/test commands will depend on the project type being developed. Common Kiro-related operations include:

- Use `#File` or `#Folder` to reference specific files/folders in chat
- Access `#Problems`, `#Terminal`, `#Git Diff` for context
- Use `#Codebase` for full codebase scanning (once indexed)
- Command palette: Search for 'MCP' or 'Kiro' commands as needed
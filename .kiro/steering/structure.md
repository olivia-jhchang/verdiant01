# Project Structure

## Directory Organization

```
.
├── .kiro/                    # Kiro AI assistant configuration
│   ├── steering/            # AI steering rules and guidelines
│   │   ├── product.md       # Product overview and purpose
│   │   ├── tech.md          # Technology stack and commands
│   │   └── structure.md     # Project organization (this file)
│   └── settings/            # Kiro-specific settings (created as needed)
│       └── mcp.json         # MCP server configurations
└── .vscode/                 # VSCode workspace configuration
    └── settings.json        # IDE settings and extensions
```

## Key Conventions

### Kiro Configuration
- All Kiro-related files live under `.kiro/`
- Steering rules are markdown files in `.kiro/steering/`
- MCP configurations go in `.kiro/settings/mcp.json`
- Steering files are automatically included in AI context

### Development Workflow
- This workspace serves as a foundation for various project types
- Project-specific files and folders will be added as development progresses
- Maintain clean separation between Kiro configuration and project code
- Use descriptive folder names and maintain consistent structure as project grows

### File Naming
- Use lowercase with hyphens for multi-word files (kebab-case)
- Keep steering document names simple and descriptive
- Follow established patterns when adding new configuration files
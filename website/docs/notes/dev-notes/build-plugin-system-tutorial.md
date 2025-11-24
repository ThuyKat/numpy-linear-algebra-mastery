# Tutorial: Build Your Own Plugin System

A hands-on guide to creating a simple but functional plugin-based blog generator from scratch.

## What We're Building

A minimal static site generator with a plugin architecture that:
- Reads markdown files
- Allows plugins to transform content
- Supports themes via plugins
- Can be extended without modifying core code

### Why Build This?

Building a plugin system teaches you:
- **Architecture design**: How to create flexible, extensible systems
- **Separation of concerns**: Keeping core logic separate from features
- **Real-world patterns**: The same patterns used in Docusaurus, Gatsby, ESLint, and Webpack
- **Scalability**: How to build systems that can grow without becoming messy

### Understanding the Final Structure

```
my-blog-generator/
├── src/
│   ├── core/                    # The engine (shouldn't change often)
│   │   ├── PluginManager.ts    # Manages plugin registration & execution
│   │   ├── Generator.ts        # Main build process & file handling
│   │   └── types.ts            # TypeScript interfaces & contracts
│   └── plugins/                 # Features (easy to add/remove)
│       ├── markdown-plugin.ts
│       ├── syntax-highlight-plugin.ts
│       └── reading-time-plugin.ts
├── content/                     # User content (markdown files)
│   └── posts/
│       └── my-first-post.md
├── config.ts                    # Configuration (which plugins to use)
└── package.json
```

#### What Each Part Does:

**`src/core/`** - The Framework
- **Purpose**: Core functionality that rarely changes
- **Contains**: Plugin management, file processing, type definitions
- **Why separate?**: You want this stable - it's the foundation
- **Think of it as**: The engine of a car

**`src/plugins/`** - The Features
- **Purpose**: Modular features that can be added/removed
- **Contains**: Individual plugins that do specific things
- **Why separate?**: Easy to add new features without touching core code
- **Think of it as**: Apps on your phone - you install what you need

**`content/`** - The Data
- **Purpose**: Your actual blog posts/pages
- **Contains**: Markdown files that users write
- **Why separate?**: Content shouldn't mix with code
- **Think of it as**: The documents you work on

**`config.ts`** - The Settings
- **Purpose**: User configuration - which plugins to enable
- **Contains**: List of plugins to use, input/output paths
- **Why needed?**: Users should control behavior without coding
- **Think of it as**: Settings app on your phone

---

## Step 1: Project Setup

**What we're doing**: Creating a new Node.js project with TypeScript support.

**Why TypeScript?**:
- Type safety catches bugs before runtime
- Better IDE autocomplete and documentation
- Essential for defining plugin interfaces clearly

### Initialize the Project

```bash
mkdir my-blog-generator
cd my-blog-generator
npm init -y
npm install typescript @types/node --save-dev
npm install marked gray-matter
```

### Setup TypeScript

Create `tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

---

## Step 2: Define Plugin Types

**What we're doing**: Creating TypeScript interfaces that define how plugins must work.

**Why this is important**:
- **Contract**: Ensures all plugins follow the same rules
- **Type safety**: Catches errors if a plugin is implemented incorrectly
- **Documentation**: The interfaces serve as documentation for plugin developers
- **Flexibility**: Plugins can implement only the hooks they need

**Think of this as**: Writing the rules of a board game - everyone needs to follow the same rules to play together.

Create `src/core/types.ts`:

```typescript
// The context object passed through the plugin chain
export interface PluginContext {
  content: string;
  metadata: Record<string, any>;
  filePath: string;
}

// Plugin interface - all plugins must implement this
export interface Plugin {
  name: string;

  // Called when plugin is registered
  init?(): void | Promise<void>;

  // Transform content before processing
  beforeProcess?(context: PluginContext): PluginContext | Promise<PluginContext>;

  // Transform content after processing
  afterProcess?(context: PluginContext): PluginContext | Promise<PluginContext>;

  // Add custom data to the final output
  enhance?(data: any): any | Promise<any>;
}

// Configuration for the generator
export interface GeneratorConfig {
  inputDir: string;
  outputDir: string;
  plugins: Plugin[];
}
```

**Key concepts explained:**

1. **PluginContext**: The "package" of data passed from plugin to plugin
   - Like a baton in a relay race - each runner (plugin) receives it, does something, and passes it on
   - Contains: original content, metadata, and file path

2. **Plugin interface**: The contract all plugins must follow
   - Like an electrical outlet - any plug (plugin) that fits the shape (interface) will work
   - Plugins can implement only the hooks they need (all are optional except `name`)

3. **Lifecycle hooks**: Different stages where plugins can intervene
   - `init()`: When the plugin first loads (setup)
   - `beforeProcess()`: Before content is converted (good for extracting data)
   - `afterProcess()`: After content is converted (good for adding features)
   - `enhance()`: Modify the final output (good for global changes)

---

## Step 3: Build the Plugin Manager

**What we're doing**: Creating the system that registers plugins and runs their hooks.

**Why we need this**:
- **Centralized control**: One place manages all plugins
- **Execution order**: Ensures plugins run in the right sequence
- **Hook coordination**: Calls the right hook at the right time
- **Separation**: Core code doesn't need to know about specific plugins

**Think of it as**: A conductor in an orchestra - tells each musician (plugin) when to play their part.

Create `src/core/PluginManager.ts`:

```typescript
import { Plugin, PluginContext } from './types';

export class PluginManager {
  private plugins: Plugin[] = [];

  // Register a plugin
  register(plugin: Plugin): void {
    console.log(`Registering plugin: ${plugin.name}`);
    this.plugins.push(plugin);

    // Call init hook if it exists
    if (plugin.init) {
      plugin.init();
    }
  }

  // Register multiple plugins
  registerAll(plugins: Plugin[]): void {
    plugins.forEach(plugin => this.register(plugin));
  }

  // Execute beforeProcess hooks
  async runBeforeProcess(context: PluginContext): Promise<PluginContext> {
    let currentContext = context;

    for (const plugin of this.plugins) {
      if (plugin.beforeProcess) {
        console.log(`Running beforeProcess: ${plugin.name}`);
        currentContext = await plugin.beforeProcess(currentContext);
      }
    }

    return currentContext;
  }

  // Execute afterProcess hooks
  async runAfterProcess(context: PluginContext): Promise<PluginContext> {
    let currentContext = context;

    for (const plugin of this.plugins) {
      if (plugin.afterProcess) {
        console.log(`Running afterProcess: ${plugin.name}`);
        currentContext = await plugin.afterProcess(currentContext);
      }
    }

    return currentContext;
  }

  // Execute enhance hooks
  async runEnhance(data: any): Promise<any> {
    let currentData = data;

    for (const plugin of this.plugins) {
      if (plugin.enhance) {
        console.log(`Running enhance: ${plugin.name}`);
        currentData = await plugin.enhance(currentData);
      }
    }

    return currentData;
  }

  // Get all registered plugins
  getPlugins(): Plugin[] {
    return this.plugins;
  }
}
```

**What this code does:**
- **`register()`**: Adds a plugin to the list and calls its `init()` hook
- **`runBeforeProcess()`**: Executes all `beforeProcess` hooks in order
- **`runAfterProcess()`**: Executes all `afterProcess` hooks in order
- **`runEnhance()`**: Executes all `enhance` hooks in order
- Each method passes the context through like a pipeline

**Important pattern**: Notice how we use `async/await` and loop through plugins sequentially. This ensures plugins run in order, not in parallel.

---

## Step 4: Create the Core Generator

**What we're doing**: Building the main engine that processes files using plugins.

**Why this is the "core"**:
- **Orchestration**: Coordinates the entire build process
- **File handling**: Reads markdown files, writes output
- **Plugin integration**: Uses PluginManager to run plugins at the right times
- **Main workflow**: Ties everything together

**The flow**:
1. Read all markdown files from input directory
2. For each file, create a PluginContext
3. Run the context through all plugin hooks
4. Collect results and write output

**Think of it as**: A factory assembly line - raw materials (markdown) come in, pass through stations (plugins), finished product (HTML) comes out.

Create `src/core/Generator.ts`:

```typescript
import * as fs from 'fs/promises';
import * as path from 'path';
import { PluginManager } from './PluginManager';
import { GeneratorConfig, PluginContext } from './types';

export class Generator {
  private pluginManager: PluginManager;
  private config: GeneratorConfig;

  constructor(config: GeneratorConfig) {
    this.config = config;
    this.pluginManager = new PluginManager();

    // Register all plugins from config
    this.pluginManager.registerAll(config.plugins);
  }

  // Main build process
  async build(): Promise<void> {
    console.log('Starting build...\n');

    // Ensure output directory exists
    await fs.mkdir(this.config.outputDir, { recursive: true });

    // Read all markdown files
    const files = await this.getMarkdownFiles(this.config.inputDir);
    console.log(`Found ${files.length} files to process\n`);

    // Process each file
    const results = [];
    for (const file of files) {
      const result = await this.processFile(file);
      results.push(result);
    }

    // Enhance final data with plugins
    const enhancedResults = await this.pluginManager.runEnhance(results);

    // Write output
    await this.writeOutput(enhancedResults);

    console.log('\n✓ Build complete!');
  }

  // Process a single file through the plugin pipeline
  private async processFile(filePath: string): Promise<any> {
    console.log(`Processing: ${filePath}`);

    // Read file content
    const rawContent = await fs.readFile(filePath, 'utf-8');

    // Create initial context
    let context: PluginContext = {
      content: rawContent,
      metadata: {},
      filePath: filePath,
    };

    // Run through beforeProcess hooks
    context = await this.pluginManager.runBeforeProcess(context);

    // Run through afterProcess hooks
    context = await this.pluginManager.runAfterProcess(context);

    return {
      ...context.metadata,
      content: context.content,
      path: filePath,
    };
  }

  // Get all markdown files from a directory
  private async getMarkdownFiles(dir: string): Promise<string[]> {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    const files: string[] = [];

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        const subFiles = await this.getMarkdownFiles(fullPath);
        files.push(...subFiles);
      } else if (entry.name.endsWith('.md')) {
        files.push(fullPath);
      }
    }

    return files;
  }

  // Write final output
  private async writeOutput(data: any): Promise<void> {
    const outputPath = path.join(this.config.outputDir, 'output.json');
    await fs.writeFile(outputPath, JSON.stringify(data, null, 2));
    console.log(`\nOutput written to: ${outputPath}`);
  }
}
```

**How it works - Key methods explained:**

1. **`build()`**: Main entry point
   - Creates output directory
   - Finds all markdown files
   - Processes each file
   - Writes final output

2. **`processFile()`**: Processes one file
   - Reads file content
   - Creates initial PluginContext
   - Runs through beforeProcess hooks
   - Runs through afterProcess hooks
   - Returns processed data

3. **`getMarkdownFiles()`**: Recursively finds all `.md` files
   - Walks through directories
   - Collects markdown file paths

**Why this design?**: The Generator doesn't know anything about markdown, HTML, or reading time. It just knows how to read files and run plugins. This separation makes it flexible!

---

## Step 5: Create Your First Plugin

**What we're doing**: Creating a plugin that parses markdown and converts it to HTML.

**Why start with this plugin?**:
- **Core functionality**: Every blog needs markdown parsing
- **Two-step process**: Shows both `beforeProcess` and `afterProcess` hooks
- **Demonstrates pattern**: Good example of how plugins should work

**What this plugin does**:
1. `beforeProcess`: Extracts metadata (frontmatter) from the top of markdown files
2. `afterProcess`: Converts markdown content to HTML

**Why two steps?**: Separating metadata extraction from HTML conversion means other plugins can access the metadata or modify the raw markdown before it becomes HTML.

Create `src/plugins/markdown-plugin.ts`:

```typescript
import { Plugin, PluginContext } from '../core/types';
import * as matter from 'gray-matter';
import { marked } from 'marked';

export const markdownPlugin: Plugin = {
  name: 'markdown-plugin',

  init() {
    console.log('Markdown plugin initialized');
  },

  async beforeProcess(context: PluginContext): Promise<PluginContext> {
    // Parse frontmatter (metadata at top of markdown files)
    const { data, content } = matter(context.content);

    return {
      ...context,
      content: content,
      metadata: {
        ...context.metadata,
        ...data,
      },
    };
  },

  async afterProcess(context: PluginContext): Promise<PluginContext> {
    // Convert markdown to HTML
    const html = await marked(context.content);

    return {
      ...context,
      content: html,
      metadata: {
        ...context.metadata,
        contentType: 'html',
      },
    };
  },
};
```

**What it does:**
- `beforeProcess`: Extracts frontmatter metadata (title, author, date, etc.)
- `afterProcess`: Converts markdown to HTML using the `marked` library

**Key insight**: This plugin doesn't care about reading time, syntax highlighting, or anything else. It does one job well. Other plugins will add other features!

---

## Step 6: Create More Plugins

**What we're doing**: Creating plugins that add extra features.

**Why multiple small plugins instead of one big one?**:
- **Modularity**: Users can enable only what they need
- **Maintainability**: Each plugin is simple and focused
- **Testability**: Easy to test individual features
- **Extensibility**: Others can add plugins without modifying yours

**The beauty of plugins**: Adding reading time or syntax highlighting doesn't require touching the core code or the markdown plugin!

### Reading Time Plugin

**What it does**: Calculates how long it takes to read a post.

**Why in `afterProcess`?**: We want to count words in the plain text, not the HTML tags, so we remove HTML first.

Create `src/plugins/reading-time-plugin.ts`:

```typescript
import { Plugin, PluginContext } from '../core/types';

export const readingTimePlugin: Plugin = {
  name: 'reading-time-plugin',

  async afterProcess(context: PluginContext): Promise<PluginContext> {
    // Count words (simple approximation)
    const text = context.content.replace(/<[^>]*>/g, ''); // Remove HTML tags
    const wordCount = text.split(/\s+/).length;

    // Average reading speed: 200 words per minute
    const readingTime = Math.ceil(wordCount / 200);

    return {
      ...context,
      metadata: {
        ...context.metadata,
        wordCount,
        readingTime: `${readingTime} min read`,
      },
    };
  },
};
```

### Syntax Highlighting Plugin

**What it does**: Adds CSS classes to code blocks for styling.

**Two hooks used**:
- `afterProcess`: Modifies HTML to add highlighting classes
- `enhance`: Adds global metadata that highlighting is available

**Why `enhance`?**: The enhance hook operates on ALL processed files at once, perfect for adding site-wide metadata.

Create `src/plugins/syntax-highlight-plugin.ts`:

```typescript
import { Plugin, PluginContext } from '../core/types';

export const syntaxHighlightPlugin: Plugin = {
  name: 'syntax-highlight-plugin',

  async afterProcess(context: PluginContext): Promise<PluginContext> {
    // Simple code block detection and wrapping
    // In a real implementation, you'd use a library like Prism or Highlight.js
    const highlightedContent = context.content.replace(
      /<code>(.*?)<\/code>/gs,
      '<code class="highlighted">$1</code>'
    );

    return {
      ...context,
      content: highlightedContent,
      metadata: {
        ...context.metadata,
        hasSyntaxHighlighting: true,
      },
    };
  },

  enhance(data: any) {
    // Add global flag that syntax highlighting is enabled
    return {
      ...data,
      siteMetadata: {
        syntaxHighlightingEnabled: true,
      },
    };
  },
};
```

---

## Step 7: Create Configuration

**What we're doing**: Creating a configuration file where users choose which plugins to use.

**Why this is powerful**:
- **User control**: Users enable/disable features without touching code
- **Plugin order matters**: Plugins run in the order listed here
- **Easy to extend**: Adding a new plugin = importing it and adding to array
- **Separation**: Configuration separate from implementation

**Think of it as**: A restaurant menu - the kitchen (core) can make anything, but diners (users) choose what they want.

**Order matters!** Notice:
1. `markdownPlugin` runs first (converts markdown)
2. `readingTimePlugin` runs second (counts words in HTML)
3. `syntaxHighlightPlugin` runs last (highlights code)

If you swapped the order, things might break! For example, if syntax highlighting ran before markdown conversion, it wouldn't find any code blocks yet.

Create `config.ts`:

```typescript
import { GeneratorConfig } from './src/core/types';
import { markdownPlugin } from './src/plugins/markdown-plugin';
import { readingTimePlugin } from './src/plugins/reading-time-plugin';
import { syntaxHighlightPlugin } from './src/plugins/syntax-highlight-plugin';

const config: GeneratorConfig = {
  inputDir: './content/posts',
  outputDir: './dist',
  plugins: [
    markdownPlugin,
    readingTimePlugin,
    syntaxHighlightPlugin,
  ],
};

export default config;
```

---

## Step 8: Create the Entry Point

**What we're doing**: Creating the main file that runs everything.

**Why so simple?**: All the complexity is hidden in the Generator and plugins. The entry point just:
1. Imports the config
2. Creates a Generator
3. Runs the build

**This is good design**: The entry point shouldn't contain business logic - just orchestration.

Create `src/index.ts`:

```typescript
import { Generator } from './core/Generator';
import config from '../config';

async function main() {
  const generator = new Generator(config);
  await generator.build();
}

main().catch(console.error);
```

---

## Step 9: Create Sample Content

**What we're doing**: Creating a sample markdown blog post to test our system.

**Anatomy of the file**:
- **Frontmatter** (between `---`): Metadata like title, author, date, tags
- **Content**: Regular markdown that will be converted to HTML

**Why frontmatter?**: Separates metadata from content. The `gray-matter` library (used in markdown-plugin) parses this for us.

Create `content/posts/my-first-post.md`:

```markdown
---
title: My First Blog Post
author: Katie
date: 2024-01-15
tags: [javascript, tutorial]
---

# Introduction to Plugin Systems

This is my first blog post about building plugin systems.

## What is a Plugin?

A plugin is a piece of code that extends the functionality of an application without modifying its core.

## Code Example

Here's a simple example:

\`\`\`javascript
const plugin = {
  name: 'my-plugin',
  init() {
    console.log('Plugin initialized!');
  }
};
\`\`\`

This architecture allows for great flexibility and extensibility.
```

---

## Step 10: Add Build Scripts

Update `package.json`:

```json
{
  "name": "my-blog-generator",
  "version": "1.0.0",
  "scripts": {
    "build": "tsc && node dist/index.js",
    "dev": "tsc && node dist/index.js --watch"
  },
  "dependencies": {
    "gray-matter": "^4.0.3",
    "marked": "^11.0.0"
  },
  "devDependencies": {
    "@types/node": "^20.10.0",
    "typescript": "^5.3.0"
  }
}
```

---

## Step 11: Run Your Generator

```bash
# Build and run
npm run build
```

**Expected output:**
```
Starting build...

Registering plugin: markdown-plugin
Markdown plugin initialized
Registering plugin: reading-time-plugin
Registering plugin: syntax-highlight-plugin
Found 1 files to process

Processing: content/posts/my-first-post.md
Running beforeProcess: markdown-plugin
Running afterProcess: markdown-plugin
Running afterProcess: reading-time-plugin
Running afterProcess: syntax-highlight-plugin
Running enhance: syntax-highlight-plugin

Output written to: dist/output.json

✓ Build complete!
```

Check `dist/output.json` to see your processed content!

---

## Understanding the Plugin Flow

```
┌─────────────────────────────────────────────────┐
│ 1. Read File: my-first-post.md                  │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 2. beforeProcess Hooks (in order)               │
│    - markdown-plugin: Extract frontmatter       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 3. afterProcess Hooks (in order)                │
│    - markdown-plugin: Convert to HTML           │
│    - reading-time-plugin: Calculate read time   │
│    - syntax-highlight-plugin: Highlight code    │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 4. enhance Hooks (on all data)                  │
│    - syntax-highlight-plugin: Add metadata      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 5. Write Output: dist/output.json               │
└─────────────────────────────────────────────────┘
```

---

## Exercise: Add Your Own Plugin

Try creating a "Table of Contents" plugin:

```typescript
// src/plugins/toc-plugin.ts
import { Plugin, PluginContext } from '../core/types';

export const tocPlugin: Plugin = {
  name: 'toc-plugin',

  async afterProcess(context: PluginContext): Promise<PluginContext> {
    // Extract all headings
    const headingRegex = /<h([1-6])>(.*?)<\/h\1>/g;
    const headings: Array<{ level: number; text: string }> = [];

    let match;
    while ((match = headingRegex.exec(context.content)) !== null) {
      headings.push({
        level: parseInt(match[1]),
        text: match[2],
      });
    }

    return {
      ...context,
      metadata: {
        ...context.metadata,
        tableOfContents: headings,
      },
    };
  },
};
```

Add it to your `config.ts` and rebuild!

---

## Putting It All Together: How Everything Works

Now that you've built the system, let's understand how all the pieces connect:

### The Big Picture

```
┌─────────────────────────────────────────────────────────┐
│                      USER RUNS                          │
│                   npm run build                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  src/index.ts                                           │
│  - Entry point                                          │
│  - Loads config.ts                                      │
│  - Creates Generator                                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  config.ts                                              │
│  - Specifies which plugins to use                       │
│  - Sets inputDir: './content/posts'                     │
│  - Sets outputDir: './dist'                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  src/core/Generator.ts                                  │
│  - Creates PluginManager                                │
│  - Registers all plugins from config                    │
│  - Reads all .md files from content/                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  src/core/PluginManager.ts                              │
│  - Holds all registered plugins                         │
│  - Executes hooks in order                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────┴──────────────┐
        │                           │
        ▼                           ▼
┌──────────────┐            ┌──────────────┐
│ For each     │            │ Plugins run  │
│ .md file:    │            │ in sequence: │
│              │            │              │
│ 1. Read file │            │ beforeProcess│
│ 2. Create    │───────────▶│ afterProcess │
│    context   │            │ enhance      │
│ 3. Run hooks │            │              │
└──────────────┘            └──────────────┘
        │                           │
        └────────────┬──────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  dist/output.json                                       │
│  - Final processed data                                 │
│  - Contains HTML, metadata, reading time, etc.          │
└─────────────────────────────────────────────────────────┘
```

### Why This Architecture is Powerful

**1. Separation of Concerns**
- **Core** (`Generator`, `PluginManager`): Handles file I/O and plugin coordination
- **Plugins**: Handle specific features (markdown, highlighting, etc.)
- **Config**: User choices
- **Content**: User's blog posts

Each part has ONE job. Changes in one area don't affect others.

**2. Open/Closed Principle**
- **Open for extension**: Add new plugins anytime
- **Closed for modification**: Core code stays stable

Want to add emoji support? Create an emoji plugin. No need to modify Generator or PluginManager!

**3. Dependency Inversion**
- Core depends on the `Plugin` interface, not specific plugins
- You can swap markdown-plugin for a different one, as long as it follows the interface
- Generator doesn't know what plugins do - it just runs their hooks

**4. Single Responsibility**
- `types.ts`: Defines contracts
- `PluginManager.ts`: Manages plugins
- `Generator.ts`: Processes files
- Each plugin: Does one thing

### What Makes This a "Plugin System"?

Compare these two approaches:

**❌ Without plugins (tightly coupled):**
```typescript
class Generator {
  processFile(file) {
    // Extract frontmatter - HARD-CODED
    const { data, content } = matter(file);

    // Convert markdown - HARD-CODED
    const html = marked(content);

    // Calculate reading time - HARD-CODED
    const readingTime = content.split(' ').length / 200;

    // Want to add more features? Modify this file!
  }
}
```

**✅ With plugins (loosely coupled):**
```typescript
class Generator {
  processFile(file) {
    let context = { content: file, metadata: {} };

    // Run whatever plugins user configured
    context = await pluginManager.runBeforeProcess(context);
    context = await pluginManager.runAfterProcess(context);

    // Want to add features? Create a plugin!
  }
}
```

The plugin version is flexible, testable, and extensible!

### Real-World Parallel: How Docusaurus Does It

Your system and Docusaurus follow the same pattern:

| Your System | Docusaurus | Purpose |
|-------------|------------|---------|
| `Generator` | `@docusaurus/core` | Core framework |
| `PluginManager` | Plugin system in core | Manages plugins |
| `markdown-plugin` | `@docusaurus/plugin-content-docs` | Content handling |
| `syntax-highlight-plugin` | `@docusaurus/theme-classic` (includes Prism) | Code highlighting |
| `config.ts` | `docusaurus.config.ts` | User configuration |

The concepts are identical, just at different scales!

---

## Key Takeaways

1. **Plugin Interface**: Define a clear contract that all plugins follow
2. **Lifecycle Hooks**: Provide multiple points where plugins can intervene
3. **Context Object**: Pass data through the plugin chain
4. **Plugin Manager**: Centralize plugin registration and execution
5. **Configuration**: Allow users to enable/disable plugins easily

## Next Steps

- Add file watching for development mode
- Create HTML template plugins
- Add plugin configuration options
- Implement plugin dependencies
- Create a CLI with commands
- Publish as an npm package

---

## Real-World Examples

Now that you understand the basics, study these production plugin systems:

- **Docusaurus**: `@docusaurus/core` with theme and preset plugins
- **Vite**: Build tool with rollup plugins
- **ESLint**: Linting rules as plugins
- **Gatsby**: Data sourcing and transformation plugins
- **Rollup**: Module bundler with extensive plugin ecosystem

The pattern you just learned is the foundation of all these systems!
# Docusaurus Architecture & Plugin Systems

## Understanding siteConfig in Docusaurus

### Where does siteConfig come from?

**Answer:** `siteConfig` comes from `docusaurus.config.ts`, NOT from `package.json`.

When you use `useDocusaurusContext()` in your React components:

```tsx
const {siteConfig} = useDocusaurusContext();
```

It retrieves the configuration object defined in your `docusaurus.config.ts` file, which includes:
- `title`: Your site title
- `tagline`: Your site tagline
- `url`, `baseUrl`: Deployment configuration
- `organizationName`, `projectName`: GitHub settings
- And many other configuration options

### How Context Gets Passed to Components

Unlike typical React apps where you manually wrap your `App.js` with context providers, **Docusaurus handles this automatically**.

#### The Hidden Setup (in `node_modules/@docusaurus/core/lib/client/App.js`)

```jsx
export default function App() {
    return (
      <ErrorBoundary>
        <DocusaurusContextProvider>  {/* ← Context Provider */}
          <BrowserContextProvider>
            <Root>
              <ThemeProvider>
                {/* Your pages render here */}
                <AppNavigation />
              </ThemeProvider>
            </Root>
          </BrowserContextProvider>
        </DocusaurusContextProvider>
      </ErrorBoundary>
    );
}
```

#### The Context Provider (in `node_modules/@docusaurus/core/lib/client/docusaurusContext.js`)

```jsx
import siteConfig from '@generated/docusaurus.config';

const contextValue = {
    siteConfig,    // Your docusaurus.config.ts data
    siteMetadata,
    globalData,
    i18n,
    codeTranslations,
};

export function DocusaurusContextProvider({ children }) {
    return <Context.Provider value={contextValue}>{children}</Context.Provider>;
}
```

#### How It Works:
1. **Build time**: Docusaurus generates `@generated/docusaurus.config` from your `docusaurus.config.ts`
2. **Automatic wrapping**: `DocusaurusContextProvider` wraps your entire app in the framework code
3. **Access anywhere**: Any component can use `useDocusaurusContext()` to access the config

---

## Modular Architecture vs Microservices

### Docusaurus Uses: Modular Monolithic Architecture

```
@docusaurus/core  ──┐
@docusaurus/theme ──┼──> Single bundled app
@docusaurus/preset ─┘    (runs in one process)
```

**Characteristics:**
- Multiple npm packages working together
- All bundled into **one application**
- Runs in a **single process** (browser/Node.js)
- Packages communicate via **direct function calls** (in-memory)
- Deployed as **one unit**
- Shares the same memory space

**Think of it as:** Building with LEGO blocks - separate pieces snap together to create **one final structure**.

### Microservices Architecture (What Docusaurus Does NOT Use)

```
Auth Service (port 3001)     ──┐
User Service (port 3002)     ──┼──> Separate processes
Payment Service (port 3003)  ──┘    communicate over network
```

**Characteristics:**
- Separate **running services** (different processes)
- Communicate over **network** (HTTP/REST/gRPC)
- Can be **deployed independently**
- Each has its own database (typically)
- Run on different servers/containers
- Loosely coupled at runtime

### Key Difference

| Aspect | Modular Monolith | Microservices |
|--------|------------------|---------------|
| **Deployment** | Single unit | Multiple independent services |
| **Communication** | Function calls | Network (HTTP/REST) |
| **Process** | One process | Multiple processes |
| **Scaling** | Scale entire app | Scale services independently |
| **Complexity** | Lower | Higher |

---

## Learning Path: Building Plugin-Based Systems

### Core Concepts to Master

#### 1. Monorepo Management
Tools to learn:
- **pnpm workspaces** (simplest to start)
- **Lerna** (classic monorepo tool)
- **Nx** (enterprise-grade)
- **Turborepo** (fast builds)

**What you'll learn:**
- Managing multiple packages in one repository
- Shared dependencies
- Cross-package imports

#### 2. Plugin Architecture Patterns
**Key concepts:**
- How to design extensible systems
- Plugin registration and lifecycle
- Event systems and hooks
- Configuration APIs

**Example simple plugin system:**
```typescript
class PluginSystem {
  private plugins: Plugin[] = [];

  register(plugin: Plugin) {
    this.plugins.push(plugin);
    plugin.init();
  }

  executeHook(hookName: string, data: any) {
    this.plugins.forEach(plugin => {
      if (plugin[hookName]) {
        plugin[hookName](data);
      }
    });
  }
}
```

#### 3. Build Tools & Bundlers
Learn one of these:
- **Vite** (modern, fast - recommended to start)
- **Webpack** (industry standard, powerful)
- **Rollup** (library bundling)
- **esbuild** (extremely fast)

**Concepts to understand:**
- Module bundling
- Code splitting
- Tree shaking
- Hot module replacement

#### 4. Package Publishing
**Skills needed:**
- Creating npm packages
- Understanding `package.json` fields
- Versioning (semver)
- Managing dependencies vs peerDependencies
- Publishing to npm registry

### Learning Resources

#### Online Courses
Search for these topics on:
- **Udemy**: "monorepo", "plugin architecture", "building frameworks", "advanced Node.js"
- **Frontend Masters**: Build tools, Advanced Node.js
- **egghead.io**: Package development, Monorepo management
- **YouTube**: "Build your own framework" series

#### Practical Learning Approach

**Phase 1: Study Existing Code**
- Read Docusaurus source on GitHub
- Study Gatsby, VitePress, or Astro
- Explore ESLint or Prettier plugin systems

**Phase 2: Build Simple Projects**
1. **Blog with widgets**: Create a blog system where plugins can add custom widgets
2. **CLI tool**: Build a command-line tool with plugin support (like ESLint)
3. **Build tool**: Create a simple bundler with plugin hooks

**Phase 3: Advanced Projects**
- Build a mini framework with routing and plugins
- Create a monorepo with multiple interconnected packages
- Implement a theme system with inheritance

### Recommended First Project: Simple Plugin System

Build a basic plugin-based blog generator:

```
my-static-site-generator/
├── packages/
│   ├── core/           # Main framework
│   ├── plugin-syntax-highlighting/
│   ├── plugin-image-optimization/
│   └── plugin-markdown/
└── package.json
```

This will teach you:
- Monorepo structure
- Plugin registration
- Hook systems
- Build pipelines

---

## Next Steps

1. Pick one monorepo tool (start with pnpm workspaces)
2. Study one existing plugin-based project deeply
3. Build a simple CLI tool with 2-3 plugins
4. Gradually increase complexity

**Remember:** The best way to learn is by building!

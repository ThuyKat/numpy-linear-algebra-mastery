# NumPy & Linear Algebra Mastery

> A 20-week structured learning journey from fundamentals to mastery in machine learning mathematics

[![Deploy to GitHub Pages](https://github.com/ThuyKat/numpy-linear-algebra-mastery/workflows/Deploy%20to%20GitHub%20Pages/badge.svg)](https://github.com/ThuyKat/numpy-linear-algebra-mastery/actions)
[![Built with Docusaurus](https://img.shields.io/badge/Built%20with-Docusaurus-blue)](https://docusaurus.io/)
[![Learning in Public](https://img.shields.io/badge/Learning%20in-Public-green)](https://www.swyx.io/learn-in-public)

## 📚 About This Project

This repository documents my comprehensive journey learning NumPy and Linear Algebra for machine learning applications. The content is organized as a learning website built with Docusaurus, featuring:

- **Structured weekly curriculum** covering 20 weeks of progressive learning
- **Hands-on projects** implementing core concepts
- **Detailed notes** on NumPy and linear algebra concepts
- **Weekly blog reflections** tracking progress and insights
- **Dev tutorials** for building similar learning platforms

### Why This Approach?

**Learning in Public**: Documenting the journey makes learning more effective through:
- Writing reinforces understanding
- Structured organization aids retention
- Sharing helps others on similar paths
- Accountability drives consistency

## 🎯 Learning Path

### Weeks 1-5: Foundations
- NumPy fundamentals (arrays, indexing, operations)
- Basic linear algebra concepts
- Vector operations and properties
- Matrix basics

### Weeks 6-10: Intermediate
- Advanced NumPy techniques
- Matrix operations and transformations
- Linear systems and solutions
- Eigenvalues and eigenvectors

### Weeks 11-15: Advanced
- Decompositions (LU, QR, SVD)
- Principal Component Analysis (PCA)
- Optimization techniques
- Numerical methods

### Weeks 16-20: Applications
- Machine learning mathematics
- Neural network foundations
- Practical ML applications
- Capstone project

## 🚀 Quick Start

### Prerequisites

- **Node.js** 20.x or higher
- **npm** or **yarn**
- **Git**

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ThuyKat/numpy-linear-algebra-mastery.git
   cd numpy-linear-algebra-mastery
   ```

2. **Install dependencies:**
   ```bash
   cd website
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000`

### Building for Production

```bash
npm run build
```

This creates an optimized production build in the `website/build` directory.

## 📂 Project Structure

```
numpy-linear-algebra-mastery/
├── .github/
│   └── workflows/
│       └── deploy.yml              # GitHub Actions deployment
├── website/                        # Docusaurus site
│   ├── docs/                       # Documentation content
│   │   ├── weekly-content/         # 20-week curriculum
│   │   ├── projects/               # Hands-on projects
│   │   ├── notes/                  # Course notes & deep dives
│   │   │   ├── dev-notes/          # Development tutorials
│   │   │   └── docusaurus-notes/   # Docusaurus guides
│   │   └── schedule/               # Synced from Confluence (optional)
│   ├── blog/                       # Weekly reflections
│   ├── src/
│   │   ├── components/             # React components
│   │   └── pages/                  # Custom pages
│   ├── static/                     # Static assets (images, etc.)
│   ├── docusaurus.config.ts        # Site configuration
│   ├── sidebars.ts                 # Navigation structure
│   └── package.json
├── scripts/                        # Utility scripts (optional)
│   ├── confluence-sync.ts          # Sync from Confluence
│   ├── confluence-api.ts
│   └── html-to-markdown.ts
├── .gitignore
└── README.md
```

### Key Directories Explained

| Directory | Purpose |
|-----------|---------|
| `docs/weekly-content/` | Main curriculum organized by weeks |
| `docs/projects/` | Practical implementation projects |
| `docs/notes/` | Reference materials and deep dives |
| `blog/` | Weekly learning reflections |
| `src/components/` | Custom React components |
| `static/` | Images, files, and other assets |

## 📖 Content Organization

### Weekly Content Structure

Each week follows a consistent format:

```markdown
---
title: Week X - Topic Name
---

## Learning Objectives
## Core Concepts
## Practical Examples
## Exercises
## Resources
```

### Project Structure

Projects build progressively:

1. **Project 1**: Matrix Operations Implementation
2. **Project 2**: Linear Transformations Visualizer
3. **Project 3**: Eigenvalue Calculator
4. **Project 4**: PCA from Scratch
5. **Capstone**: End-to-end ML Application

## 🛠️ Technologies Used

### Core Technologies
- **[Docusaurus 3](https://docusaurus.io/)** - Static site generator
- **[React](https://react.dev/)** - UI framework
- **[TypeScript](https://www.typescriptlang.org/)** - Type-safe JavaScript
- **[MDX](https://mdxjs.com/)** - Markdown with React components

### Development Tools
- **[GitHub Actions](https://github.com/features/actions)** - CI/CD automation
- **[GitHub Pages](https://pages.github.com/)** - Free hosting
- **[Node.js](https://nodejs.org/)** - JavaScript runtime
- **[npm](https://www.npmjs.com/)** - Package manager

### Optional Integrations
- **[Confluence API](https://developer.atlassian.com/cloud/confluence/rest/)** - Content synchronization
- **[Prettier](https://prettier.io/)** - Code formatting
- **[ESLint](https://eslint.org/)** - Code linting

## 🔧 Configuration

### Site Configuration

Edit `website/docusaurus.config.ts`:

```typescript
const config: Config = {
  title: 'NumPy & Linear Algebra Mastery',
  tagline: 'My journey from fundamentals to mastery',
  url: 'https://thuykat.github.io',
  baseUrl: '/numpy-linear-algebra-mastery/',
  organizationName: 'ThuyKat',
  projectName: 'numpy-linear-algebra-mastery',
  // ... more config
};
```

### Sidebar Navigation

Edit `website/sidebars.ts`:

```typescript
const sidebars: SidebarsConfig = {
  weeklyContentSidebar: [
    // Weekly curriculum
  ],
  projectsSidebar: [
    // Projects
  ],
  notesSidebar: [
    // Notes and references
  ],
};
```

## 📝 Adding Content

### Create a New Week

1. Create file: `docs/weekly-content/week-XX.md`
2. Add frontmatter:
   ```markdown
   ---
   title: Week XX - Topic Name
   sidebar_label: Week XX
   ---
   ```
3. Add to sidebar in `sidebars.ts`

### Create a New Project

1. Create file: `docs/projects/project-XX-name.md`
2. Include: objectives, implementation, code examples
3. Add to projects sidebar

### Write a Blog Post

```bash
cd website
npm run docusaurus docs:version 1.0  # Create versioned docs (optional)
```

Create `blog/YYYY-MM-DD-post-title.md`:

```markdown
---
slug: week-1-reflections
title: Week 1 - Starting the Journey
authors: [katie]
tags: [numpy, learning, week-1]
---

Your content here...
```

## 🚢 Deployment

### Automatic Deployment (GitHub Actions)

Every push to `main` branch automatically:
1. Builds the site
2. Deploys to GitHub Pages
3. Available at: `https://thuykat.github.io/numpy-linear-algebra-mastery/`

### Manual Deployment

```bash
cd website
npm run build

# Deploy using GitHub Pages
GIT_USER=<Your GitHub username> npm run deploy
```

### First-Time Setup

1. **Enable GitHub Pages:**
   - Go to repo **Settings** → **Pages**
   - Source: **GitHub Actions**

2. **Configure base URL:**
   - Update `baseUrl` in `docusaurus.config.ts`
   - Format: `/repository-name/`

3. **Push to main:**
   ```bash
   git push origin main
   ```

4. **Check deployment:**
   - Go to **Actions** tab
   - Wait for workflow to complete
   - Visit your site!

## 🔄 Optional: Confluence Integration

Sync schedule and tracking from Confluence:

### Setup

1. **Create `.env` file:**
   ```bash
   CONFLUENCE_BASE_URL=https://your-domain.atlassian.net
   CONFLUENCE_EMAIL=your-email@example.com
   CONFLUENCE_API_TOKEN=your-token
   CONFLUENCE_PAGE_IDS=123456789,987654321
   ```

2. **Run sync:**
   ```bash
   npm run sync
   ```

See [Confluence API Integration Tutorial](website/docs/notes/dev-notes/confluence-api-integration.md) for details.

## 📚 Dev Tutorials Included

This repository includes comprehensive tutorials on:

1. **[Docusaurus Architecture](website/docs/notes/dev-notes/docusaurus-architecture.md)**
   - How siteConfig and context work
   - Modular vs microservices architecture

2. **[Build a Plugin System](website/docs/notes/dev-notes/build-plugin-system-tutorial.md)**
   - Step-by-step tutorial
   - Learn extensible architecture patterns

3. **[Confluence API Integration](website/docs/notes/dev-notes/confluence-api-integration.md)**
   - Sync external content automatically
   - Complete working code examples

4. **[GitHub Actions Guide](website/docs/notes/dev-notes/github-actions-guide.md)**
   - Understand CI/CD workflows
   - Write your own automation

## 🤝 Contributing

This is a personal learning project, but suggestions are welcome!

### Reporting Issues

Found a typo or error? [Open an issue](https://github.com/ThuyKat/numpy-linear-algebra-mastery/issues)

### Suggesting Improvements

Have ideas for better explanations or additional content? Feel free to:
- Open an issue with your suggestion
- Fork and submit a pull request
- Reach out via [discussions](https://github.com/ThuyKat/numpy-linear-algebra-mastery/discussions)

## 📊 Progress Tracking

- **Start Date**: [Your start date]
- **Current Week**: [Current progress]
- **Completion Target**: [Target date]

Track detailed progress in the [Schedule & Tracking](website/docs/schedule/) section.

## 🎓 Learning Resources

### Recommended Books
- *Linear Algebra Done Right* by Sheldon Axler
- *Introduction to Linear Algebra* by Gilbert Strang
- *Python for Data Analysis* by Wes McKinney

### Online Courses
- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [MIT OpenCourseWare - Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
- [NumPy Official Tutorials](https://numpy.org/doc/stable/user/absolute_beginners.html)

### Documentation
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [Docusaurus Documentation](https://docusaurus.io/docs)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Katie**
- GitHub: [@ThuyKat](https://github.com/ThuyKat)
- Learning Journey: [Blog Posts](website/blog/)

## 🌟 Acknowledgments

- **Docusaurus Team** - For the amazing documentation platform
- **NumPy Community** - For comprehensive documentation
- **Learn in Public Movement** - For inspiration to document this journey

## 📮 Contact & Feedback

- **Issues**: [GitHub Issues](https://github.com/ThuyKat/numpy-linear-algebra-mastery/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ThuyKat/numpy-linear-algebra-mastery/discussions)

---

## 🚀 Quick Commands Reference

```bash
# Development
npm start                 # Start dev server
npm run build            # Build for production
npm run serve            # Preview production build locally

# Content Management
npm run sync             # Sync from Confluence (if configured)
npm run clear            # Clear cache

# Deployment
npm run deploy           # Manual deploy to GitHub Pages

# Code Quality
npm run lint             # Run linter
npm run format           # Format code with Prettier
```

## 💡 Tips for Learners

1. **Consistency over intensity** - Better to study 30 minutes daily than 5 hours once a week
2. **Code along** - Don't just read, implement every example
3. **Reflect weekly** - Use the blog to solidify learning
4. **Connect concepts** - Link new learning to previous weeks
5. **Build projects** - Apply knowledge in practical scenarios

---

<div align="center">

**[📖 Start Learning](https://thuykat.github.io/numpy-linear-algebra-mastery/docs/weekly-content/introduction)** |
**[🎯 View Projects](https://thuykat.github.io/numpy-linear-algebra-mastery/docs/projects/project-01-matrix-operations)** |
**[📝 Read Blog](https://thuykat.github.io/numpy-linear-algebra-mastery/blog)**

Made with ❤️ and lots of ☕ while learning in public

</div>

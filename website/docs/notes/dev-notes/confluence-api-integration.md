# Tutorial: Sync Confluence Pages to Docusaurus

A step-by-step guide to automatically pull content from Confluence and display it in your Docusaurus site.

## What We're Building

An automated system that:
- Fetches pages from Confluence using REST API
- Converts Confluence storage format (HTML) to Markdown
- Saves the content to your Docusaurus docs
- Runs automatically before each build

**Use case**: Keep your schedule and tracking docs in Confluence (where your team works) while automatically syncing them to your learning site.

### Why This Approach?

**Benefits:**
- **Single source of truth**: Update in Confluence, auto-sync to website
- **Team collaboration**: Others can update Confluence, changes appear on site
- **Version control**: Synced content is committed to git
- **Automation**: No manual copy-paste needed

**When to use:**
- You already manage content in Confluence
- Content is updated frequently
- Multiple people contribute to the content
- You want to keep documentation in a familiar tool

---

## Prerequisites

Before starting, you'll need:
1. **Confluence Cloud account** (or Confluence Server)
2. **API token** from Confluence
3. **Node.js** installed (you already have this for Docusaurus)
4. **Page IDs** of the Confluence pages you want to sync

---

## Step 1: Get Your Confluence Credentials

### Create an API Token

**For Confluence Cloud:**

1. Go to: https://id.atlassian.com/manage-profile/security/api-tokens
2. Click "Create API token"
3. Give it a name: "Docusaurus Sync"
4. Copy the token (you won't see it again!)

**What you'll need:**
- `CONFLUENCE_EMAIL`: Your Atlassian account email
- `CONFLUENCE_API_TOKEN`: The token you just created
- `CONFLUENCE_BASE_URL`: Your Confluence URL (e.g., `https://your-domain.atlassian.net`)

### Find Your Page IDs

**Method 1: From URL**
When viewing a page, look at the URL:
```
https://your-domain.atlassian.net/wiki/spaces/SPACE/pages/123456789/Page+Title
                                                              ^^^^^^^^^
                                                              This is the page ID
```

**Method 2: From API**
```bash
curl -u your-email@example.com:your-api-token \
  "https://your-domain.atlassian.net/wiki/rest/api/content?title=Your+Page+Title&spaceKey=YOUR_SPACE"
```

**Store these safely** - we'll use environment variables to keep them secure.

---

## Step 2: Project Setup

### Install Required Packages

In your website directory:

```bash
cd website
npm install --save-dev node-fetch turndown dotenv cheerio
npm install --save-dev @types/node @types/cheerio
```

**What each package does:**
- `node-fetch`: Make HTTP requests to Confluence API
- `turndown`: Convert HTML to Markdown
- `dotenv`: Load environment variables from .env file
- `cheerio`: Parse and manipulate HTML (for cleanup)

### Create Environment File

Create `website/.env`:

```bash
# Confluence API Credentials
CONFLUENCE_BASE_URL=https://your-domain.atlassian.net
CONFLUENCE_EMAIL=your-email@example.com
CONFLUENCE_API_TOKEN=your-api-token-here

# Pages to sync (comma-separated page IDs)
CONFLUENCE_PAGE_IDS=123456789,987654321
```

⚠️ **Important**: Add `.env` to your `.gitignore` to keep credentials safe!

```bash
echo ".env" >> .gitignore
```

---

## Step 3: Create the Sync Script

### Directory Structure

```
website/
├── scripts/
│   ├── confluence-sync.ts
│   ├── confluence-api.ts
│   └── html-to-markdown.ts
├── docs/
│   └── schedule/           # Synced content goes here
├── .env
└── package.json
```

### Create the API Client

Create `scripts/confluence-api.ts`:

```typescript
import fetch from 'node-fetch';

interface ConfluenceConfig {
  baseUrl: string;
  email: string;
  apiToken: string;
}

interface ConfluencePage {
  id: string;
  title: string;
  body: {
    storage: {
      value: string; // HTML content
    };
  };
  version: {
    number: number;
  };
  _links: {
    webui: string;
  };
}

export class ConfluenceAPI {
  private config: ConfluenceConfig;
  private authHeader: string;

  constructor(config: ConfluenceConfig) {
    this.config = config;
    // Create base64 encoded auth header
    const auth = Buffer.from(
      `${config.email}:${config.apiToken}`
    ).toString('base64');
    this.authHeader = `Basic ${auth}`;
  }

  /**
   * Fetch a single page by ID
   */
  async getPage(pageId: string): Promise<ConfluencePage> {
    const url = `${this.config.baseUrl}/wiki/rest/api/content/${pageId}?expand=body.storage,version`;

    console.log(`Fetching page: ${pageId}`);

    const response = await fetch(url, {
      headers: {
        'Authorization': this.authHeader,
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(
        `Failed to fetch page ${pageId}: ${response.status} ${response.statusText}`
      );
    }

    return await response.json() as ConfluencePage;
  }

  /**
   * Fetch multiple pages
   */
  async getPages(pageIds: string[]): Promise<ConfluencePage[]> {
    const pages: ConfluencePage[] = [];

    for (const pageId of pageIds) {
      try {
        const page = await this.getPage(pageId);
        pages.push(page);
      } catch (error) {
        console.error(`Error fetching page ${pageId}:`, error);
      }
    }

    return pages;
  }

  /**
   * Search for pages by title
   */
  async searchPages(spaceKey: string, title: string): Promise<ConfluencePage[]> {
    const url = `${this.config.baseUrl}/wiki/rest/api/content?spaceKey=${spaceKey}&title=${encodeURIComponent(title)}&expand=body.storage,version`;

    const response = await fetch(url, {
      headers: {
        'Authorization': this.authHeader,
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Search failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json() as { results: ConfluencePage[] };
    return data.results;
  }
}
```

**What this does:**
- **Authentication**: Creates Basic Auth header from email + token
- **getPage()**: Fetches a single page with its content
- **getPages()**: Fetches multiple pages (for bulk sync)
- **searchPages()**: Find pages by title (alternative to using IDs)

---

## Step 4: Create HTML to Markdown Converter

Create `scripts/html-to-markdown.ts`:

```typescript
import TurndownService from 'turndown';
import * as cheerio from 'cheerio';

export class HTMLToMarkdownConverter {
  private turndown: TurndownService;

  constructor() {
    this.turndown = new TurndownService({
      headingStyle: 'atx',
      codeBlockStyle: 'fenced',
      bulletListMarker: '-',
    });

    // Add custom rules for Confluence-specific elements
    this.setupCustomRules();
  }

  /**
   * Convert Confluence HTML to clean Markdown
   */
  convert(html: string, pageTitle: string): string {
    // Clean up Confluence HTML
    const cleanedHtml = this.cleanConfluenceHTML(html);

    // Convert to markdown
    let markdown = this.turndown.turndown(cleanedHtml);

    // Add frontmatter
    markdown = this.addFrontmatter(markdown, pageTitle);

    // Clean up markdown
    markdown = this.cleanMarkdown(markdown);

    return markdown;
  }

  /**
   * Clean Confluence-specific HTML elements
   */
  private cleanConfluenceHTML(html: string): string {
    const $ = cheerio.load(html);

    // Remove Confluence macros that don't translate well
    $('.confluence-information-macro').remove();
    $('.confluence-embedded-file').remove();

    // Convert Confluence code blocks
    $('ac\\:structured-macro[ac\\:name="code"]').each((_, elem) => {
      const language = $(elem).find('ac\\:parameter[ac\\:name="language"]').text();
      const code = $(elem).find('ac\\:plain-text-body').text();

      $(elem).replaceWith(
        `<pre><code class="language-${language}">${code}</code></pre>`
      );
    });

    // Convert Confluence panels to admonitions
    $('.panel').each((_, elem) => {
      const title = $(elem).find('.panelHeader').text();
      const content = $(elem).find('.panelContent').html();

      $(elem).replaceWith(
        `<div class="admonition note">
          <div class="admonition-title">${title}</div>
          <div>${content}</div>
        </div>`
      );
    });

    // Convert status macros
    $('ac\\:structured-macro[ac\\:name="status"]').each((_, elem) => {
      const status = $(elem).find('ac\\:parameter').text();
      $(elem).replaceWith(`**Status: ${status}**`);
    });

    return $.html();
  }

  /**
   * Add Docusaurus frontmatter
   */
  private addFrontmatter(markdown: string, pageTitle: string): string {
    const frontmatter = `---
title: ${pageTitle}
sidebar_label: ${pageTitle}
synced_from_confluence: true
last_synced: ${new Date().toISOString()}
---

`;
    return frontmatter + markdown;
  }

  /**
   * Clean up markdown formatting issues
   */
  private cleanMarkdown(markdown: string): string {
    // Remove excessive newlines
    markdown = markdown.replace(/\n{3,}/g, '\n\n');

    // Fix list formatting
    markdown = markdown.replace(/^(\s*)-\s+/gm, '- ');

    // Remove empty links
    markdown = markdown.replace(/\[]\(\)/g, '');

    return markdown.trim();
  }

  /**
   * Setup custom Turndown rules
   */
  private setupCustomRules(): void {
    // Rule for handling tables
    this.turndown.addRule('confluenceTables', {
      filter: 'table',
      replacement: (content, node) => {
        // Preserve table structure
        return '\n\n' + node.outerHTML + '\n\n';
      }
    });

    // Rule for handling Confluence task lists
    this.turndown.addRule('taskList', {
      filter: (node) => {
        return (
          node.nodeName === 'LI' &&
          node.classList?.contains('task-list-item')
        );
      },
      replacement: (content, node: any) => {
        const checked = node.classList.contains('checked');
        return `- [${checked ? 'x' : ' '}] ${content}\n`;
      }
    });
  }
}
```

**What this does:**
- **cleanConfluenceHTML()**: Removes/converts Confluence-specific elements
- **addFrontmatter()**: Adds metadata for Docusaurus
- **cleanMarkdown()**: Fixes formatting issues
- **Custom rules**: Handles tables, task lists, code blocks

**Why these conversions?**
- Confluence uses custom HTML macros that need translation
- Status badges → Bold text
- Panels → Admonitions (note/warning boxes)
- Code macros → Markdown code blocks

---

## Step 5: Create the Main Sync Script

Create `scripts/confluence-sync.ts`:

```typescript
import * as fs from 'fs/promises';
import * as path from 'path';
import * as dotenv from 'dotenv';
import { ConfluenceAPI } from './confluence-api';
import { HTMLToMarkdownConverter } from './html-to-markdown';

// Load environment variables
dotenv.config();

interface SyncConfig {
  baseUrl: string;
  email: string;
  apiToken: string;
  pageIds: string[];
  outputDir: string;
}

class ConfluenceSync {
  private api: ConfluenceAPI;
  private converter: HTMLToMarkdownConverter;
  private config: SyncConfig;

  constructor(config: SyncConfig) {
    this.config = config;
    this.api = new ConfluenceAPI({
      baseUrl: config.baseUrl,
      email: config.email,
      apiToken: config.apiToken,
    });
    this.converter = new HTMLToMarkdownConverter();
  }

  /**
   * Main sync function
   */
  async sync(): Promise<void> {
    console.log('🔄 Starting Confluence sync...\n');

    try {
      // Ensure output directory exists
      await fs.mkdir(this.config.outputDir, { recursive: true });

      // Fetch all pages
      console.log(`📥 Fetching ${this.config.pageIds.length} pages from Confluence...`);
      const pages = await this.api.getPages(this.config.pageIds);

      console.log(`✅ Fetched ${pages.length} pages\n`);

      // Convert and save each page
      for (const page of pages) {
        await this.processPage(page);
      }

      // Create index file
      await this.createIndex(pages);

      console.log('\n✨ Sync complete!');
    } catch (error) {
      console.error('❌ Sync failed:', error);
      throw error;
    }
  }

  /**
   * Process a single page
   */
  private async processPage(page: any): Promise<void> {
    console.log(`📝 Processing: ${page.title}`);

    // Convert HTML to Markdown
    const markdown = this.converter.convert(
      page.body.storage.value,
      page.title
    );

    // Create safe filename
    const filename = this.createFilename(page.title);
    const filepath = path.join(this.config.outputDir, filename);

    // Write to file
    await fs.writeFile(filepath, markdown, 'utf-8');

    console.log(`   ✅ Saved to: ${filepath}`);
  }

  /**
   * Create an index file listing all synced pages
   */
  private async createIndex(pages: any[]): Promise<void> {
    const indexContent = `---
title: Confluence Pages
sidebar_label: Overview
---

# Synced from Confluence

Last synced: ${new Date().toLocaleString()}

## Pages

${pages.map(page => `- [${page.title}](./${this.createFilename(page.title).replace('.md', '')})`).join('\n')}

---

*These pages are automatically synced from Confluence. To update, edit in Confluence and run the sync script.*
`;

    const indexPath = path.join(this.config.outputDir, 'index.md');
    await fs.writeFile(indexPath, indexContent, 'utf-8');
    console.log(`📋 Created index: ${indexPath}`);
  }

  /**
   * Create a safe filename from page title
   */
  private createFilename(title: string): string {
    return title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '') + '.md';
  }
}

/**
 * Main execution
 */
async function main() {
  // Validate environment variables
  const baseUrl = process.env.CONFLUENCE_BASE_URL;
  const email = process.env.CONFLUENCE_EMAIL;
  const apiToken = process.env.CONFLUENCE_API_TOKEN;
  const pageIdsStr = process.env.CONFLUENCE_PAGE_IDS;

  if (!baseUrl || !email || !apiToken || !pageIdsStr) {
    console.error('❌ Missing required environment variables!');
    console.error('Required: CONFLUENCE_BASE_URL, CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN, CONFLUENCE_PAGE_IDS');
    process.exit(1);
  }

  const pageIds = pageIdsStr.split(',').map(id => id.trim());

  // Create sync instance
  const sync = new ConfluenceSync({
    baseUrl,
    email,
    apiToken,
    pageIds,
    outputDir: path.join(__dirname, '../docs/schedule'),
  });

  // Run sync
  await sync.sync();
}

// Run if executed directly
if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { ConfluenceSync };
```

**What this does:**
1. Loads credentials from `.env`
2. Fetches specified pages from Confluence
3. Converts each page to markdown
4. Saves to `docs/schedule/`
5. Creates an index page listing all synced pages

---

## Step 6: Update TypeScript Configuration

Update `website/tsconfig.json` to include scripts:

```json
{
  "extends": "@docusaurus/tsconfig",
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@site/*": ["./*"]
    },
    "resolveJsonModule": true,
    "esModuleInterop": true
  },
  "include": [
    "src/**/*",
    "scripts/**/*"  // Add this line
  ]
}
```

---

## Step 7: Add NPM Scripts

Update `website/package.json`:

```json
{
  "scripts": {
    "docusaurus": "docusaurus",
    "start": "docusaurus start",
    "build": "npm run sync:confluence && docusaurus build",
    "sync:confluence": "ts-node scripts/confluence-sync.ts",
    "sync": "npm run sync:confluence"
  }
}
```

**What these do:**
- `npm run sync`: Manually sync Confluence pages
- `npm run build`: Syncs before building (ensures latest content)

---

## Step 8: Update Sidebar Configuration

Add the synced content to your sidebar in `sidebars.ts`:

```typescript
notesSidebar: [
  // ... existing items ...
  {
    type: 'category',
    label: 'Schedule & Tracking',
    items: [
      {
        type: 'autogenerated',
        dirName: 'schedule',
      },
    ],
  },
],
```

**Why autogenerated?**: As you add more pages to Confluence, they'll automatically appear in the sidebar after sync.

---

## Step 9: Run Your First Sync

```bash
# Make sure you're in the website directory
cd website

# Run the sync
npm run sync
```

**Expected output:**
```
🔄 Starting Confluence sync...

📥 Fetching 2 pages from Confluence...
✅ Fetched 2 pages

📝 Processing: Week 1-5 Schedule
   ✅ Saved to: docs/schedule/week-1-5-schedule.md
📝 Processing: Learning Tracker
   ✅ Saved to: docs/schedule/learning-tracker.md
📋 Created index: docs/schedule/index.md

✨ Sync complete!
```

Check `docs/schedule/` - your Confluence pages are now there as markdown!

---

## Step 10: Test Your Site

```bash
npm start
```

Navigate to **Notes → Schedule & Tracking** - you should see your synced Confluence pages!

---

## Advanced: Automation

### Option 1: Pre-build Hook

Already done! The build script runs sync automatically.

### Option 2: GitHub Actions (Auto-sync on schedule)

Create `.github/workflows/sync-confluence.yml`:

```yaml
name: Sync Confluence

on:
  schedule:
    # Run every day at 6 AM UTC
    - cron: '0 6 * * *'
  workflow_dispatch: # Allow manual trigger

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: |
          cd website
          npm install

      - name: Sync from Confluence
        env:
          CONFLUENCE_BASE_URL: ${{ secrets.CONFLUENCE_BASE_URL }}
          CONFLUENCE_EMAIL: ${{ secrets.CONFLUENCE_EMAIL }}
          CONFLUENCE_API_TOKEN: ${{ secrets.CONFLUENCE_API_TOKEN }}
          CONFLUENCE_PAGE_IDS: ${{ secrets.CONFLUENCE_PAGE_IDS }}
        run: |
          cd website
          npm run sync

      - name: Commit changes
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add website/docs/schedule/
          git diff --quiet && git diff --staged --quiet || git commit -m "chore: sync from Confluence [skip ci]"
          git push
```

**Setup:**
1. Go to GitHub repo → Settings → Secrets
2. Add secrets: `CONFLUENCE_BASE_URL`, `CONFLUENCE_EMAIL`, `CONFLUENCE_API_TOKEN`, `CONFLUENCE_PAGE_IDS`
3. Content syncs daily automatically!

### Option 3: Webhook (Real-time sync)

For instant updates when Confluence changes, set up a webhook endpoint. (More advanced - let me know if you want this tutorial!)

---

## Troubleshooting

### Error: "Failed to fetch page"

**Cause**: Wrong page ID or authentication issue

**Fix:**
1. Verify page ID in Confluence URL
2. Check API token hasn't expired
3. Ensure email matches Atlassian account

### Error: "Cannot find module 'node-fetch'"

**Cause**: Missing dependencies

**Fix:**
```bash
npm install node-fetch turndown dotenv cheerio
```

### Content looks wrong after conversion

**Cause**: Confluence uses custom macros

**Fix:** Add custom conversion rules in `html-to-markdown.ts`. Check the Confluence HTML in the API response to see what elements need handling.

### Sync is slow

**Cause**: Fetching pages sequentially

**Optimization:** Fetch pages in parallel:

```typescript
const pages = await Promise.all(
  pageIds.map(id => this.api.getPage(id))
);
```

---

## Best Practices

1. **Commit synced content**: Include in version control for history
2. **Run sync before deploy**: Ensure latest content is published
3. **Test locally first**: Always run `npm run sync` locally before pushing
4. **Document page IDs**: Keep a list of synced pages in README
5. **Use descriptive titles**: Helps with automatic filename generation
6. **Clean up old pages**: Remove from Confluence → rerun sync → commit deletion

---

## Next Steps

**Enhancements you can add:**

1. **Incremental sync**: Only fetch pages changed since last sync
2. **Image handling**: Download and reference Confluence images
3. **Attachment sync**: Pull PDF/Excel attachments
4. **Bi-directional sync**: Update Confluence from Docusaurus (advanced!)
5. **Multi-space support**: Sync from multiple Confluence spaces
6. **Conflict detection**: Warn if page was modified in both places

---

## Summary

You now have a complete Confluence → Docusaurus pipeline!

**What you built:**
- ✅ Confluence API client with authentication
- ✅ HTML to Markdown converter
- ✅ Automated sync script
- ✅ Integration with Docusaurus build process
- ✅ (Optional) GitHub Actions for daily sync

**Your workflow now:**
1. Update schedule in Confluence (where you already work)
2. Run `npm run sync` (or let GitHub Actions do it)
3. Changes appear on your website automatically!

No more copy-paste, no manual updates, single source of truth! 🎉
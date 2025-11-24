import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  //Katie changed this
  title: 'NumPy & Linear Algebra Mastery',
  tagline: 'My journey from fundamentals to mastery in machine learning mathematics',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // NEED TO REPLACE THIS WITH GITHUB PAGES URL WHEN DEPLOYING
  url: 'https://your-docusaurus-site.example.com',
  // Katie changed this
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/numpy-linear-algebra-mastery/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'ThuyKat', // Usually your GitHub org/user name.
  projectName: 'numpy-linear-algebra-mastery', // Usually your repo name.

  onBrokenLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Katie changed this 
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/YOUR_USERNAME/numpy-linear-algebra-mastery/tree/main/website/',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
          //extra Katie added
          blogTitle: 'Learning Journey',
          blogDescription: 'Weekly reflections and insights',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'NumPy & LA Mastery',
      logo: {
        alt: 'My Site Logo',
        src: 'img/logo.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'weeklyContentSidebar',
          position: 'left',
          label: 'Weekly Content',
        },
        {
          type: 'docSidebar',
          sidebarId: 'projectsSidebar',
          position: 'left',
          label: 'Projects',
        },
        {
          type: 'docSidebar',
          sidebarId: 'notesSidebar',
          position: 'left',
          label: 'Notes',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
        {
          href: 'https://github.com/ThuyKat/numpy-linear-algebra-mastery',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
  style: 'dark',
  links: [
    {
      title: 'Learn',
      items: [
        {
          label: 'Weekly Content',
          to: '/docs/weekly-content/introduction',
        },
        {
          label: 'Projects',
          to: '/docs/projects/overview',
        },
        {
          label: 'Notes',
          to: '/docs/notes/overview',
        },
      ],
    },
    {
      title: 'Resources',
      items: [
        {
          label: 'Coursera - Deep Learning',
          href: 'https://www.coursera.org/specializations/deep-learning',
        },
        {
          label: 'NumPy Documentation',
          href: 'https://numpy.org/doc/',
        },
      ],
    },
    {
      title: 'More',
      items: [
        {
          label: 'Blog',
          to: '/blog',
        },
        {
          label: 'GitHub',
          href: 'https://github.com/YOUR_USERNAME/numpy-linear-algebra-mastery',
        },
      ],
    },
  ],
  copyright: `Learning in public © ${new Date().getFullYear()} ThuyKat`,
},
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

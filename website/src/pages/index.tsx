import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';

function HeroSection() {
  const {siteConfig} = useDocusaurusContext();
  
  return (
    <section className={styles.hero}>
      <div className={styles.heroContent}>
        <div className={styles.heroText}>
          <span className={styles.badge}>20-Week Journey</span>
          <h1 className={styles.heroTitle}>
            Master <span className={styles.highlight}>NumPy</span> &<br />
            <span className={styles.highlight}>Linear Algebra</span>
          </h1>
          <p className={styles.heroSubtitle}>
            From fundamentals to machine learning mathematics. 
            Build the foundation for your FAANG career with hands-on projects and weekly deep dives.
          </p>
          <div className={styles.heroButtons}>
            <Link className={styles.primaryButton} to="/docs/weekly-content/introduction">
              Start Learning →
            </Link>
            <Link className={styles.secondaryButton} to="/docs/projects/overview">
              View Projects
            </Link>
          </div>
          <div className={styles.stats}>
            <div className={styles.stat}>
              <span className={styles.statNumber}>20</span>
              <span className={styles.statLabel}>Weeks</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statNumber}>15+</span>
              <span className={styles.statLabel}>Projects</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statNumber}>200+</span>
              <span className={styles.statLabel}>Hours</span>
            </div>
          </div>
        </div>
        <div className={styles.heroVisual}>
          <div className={styles.cubeContainer}>
            <div className={styles.cube}>
              <div className={`${styles.face} ${styles.front}`}></div>
              <div className={`${styles.face} ${styles.back}`}></div>
              <div className={`${styles.face} ${styles.right}`}></div>
              <div className={`${styles.face} ${styles.left}`}></div>
              <div className={`${styles.face} ${styles.top}`}></div>
              <div className={`${styles.face} ${styles.bottom}`}></div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function LearningPath() {
  const phases = [
    {
      phase: "Phase 1",
      weeks: "Weeks 1-5",
      title: "Foundations",
      topics: ["NumPy Basics", "Matrix Operations", "Linear Systems"],
      color: "#4D77CF"
    },
    {
      phase: "Phase 2",
      weeks: "Weeks 6-10",
      title: "Core Concepts",
      topics: ["Eigenvalues", "SVD", "Decompositions"],
      color: "#FFD43B"
    },
    {
      phase: "Phase 3",
      weeks: "Weeks 11-15",
      title: "Applications",
      topics: ["PCA", "Optimization", "Neural Networks"],
      color: "#4B0082"
    },
    {
      phase: "Phase 4",
      weeks: "Weeks 16-20",
      title: "Advanced ML",
      topics: ["CNNs", "RNNs", "Transformers"],
      color: "#FF7F50"
    }
  ];

  return (
    <section className={styles.learningPath}>
      <div className={styles.container}>
        <h2 className={styles.sectionTitle}>Your Learning Journey</h2>
        <p className={styles.sectionSubtitle}>
          Structured progression from fundamentals to advanced machine learning
        </p>
        <div className={styles.pathGrid}>
          {phases.map((phase, idx) => (
            <div key={idx} className={styles.phaseCard} style={{'--phase-color': phase.color} as React.CSSProperties}>
              <div className={styles.phaseNumber}>{phase.phase}</div>
              <div className={styles.phaseWeeks}>{phase.weeks}</div>
              <h3 className={styles.phaseTitle}>{phase.title}</h3>
              <ul className={styles.topicList}>
                {phase.topics.map((topic, i) => (
                  <li key={i}>{topic}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Features() {
  const features = [
    {
      icon: "📊",
      title: "Real Projects",
      description: "Build recommendation systems, image compression, and NLP tools from scratch"
    },
    {
      icon: "💻",
      title: "Code-First Learning",
      description: "Every concept implemented in NumPy with detailed explanations"
    },
    {
      icon: "📝",
      title: "Comprehensive Notes",
      description: "Detailed documentation from Coursera courses and hands-on practice"
    },
    {
      icon: "🎯",
      title: "Interview Ready",
      description: "Build portfolio projects that impress FAANG recruiters"
    }
  ];

  return (
    <section className={styles.features}>
      <div className={styles.container}>
        <div className={styles.featureGrid}>
          {features.map((feature, idx) => (
            <div key={idx} className={styles.featureCard}>
              <div className={styles.featureIcon}>{feature.icon}</div>
              <h3 className={styles.featureTitle}>{feature.title}</h3>
              <p className={styles.featureDescription}>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CTA() {
  return (
    <section className={styles.cta}>
      <div className={styles.container}>
        <div className={styles.ctaContent}>
          <h2 className={styles.ctaTitle}>Ready to Master ML Mathematics?</h2>
          <p className={styles.ctaSubtitle}>
            Join the journey from NumPy basics to implementing neural networks from scratch
          </p>
          <Link className={styles.ctaButton} to="/docs/weekly-content/introduction">
            Begin Week 1 →
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Home"
      description="20-week journey mastering NumPy and Linear Algebra for Machine Learning">
      <HeroSection />
      <Features />
      <LearningPath />
      <CTA />
    </Layout>
  );
}

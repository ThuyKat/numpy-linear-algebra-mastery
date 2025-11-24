import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Structured Weekly Learning',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Follow a 20-week curriculum covering NumPy fundamentals, linear algebra concepts,
        and their applications in machine learning. Each week builds on the last,
        creating a solid foundation in ML mathematics.
      </>
    ),
  },
  {
    title: 'Hands-On Projects',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Learn by doing with practical projects that implement matrix operations,
        linear transformations, and real-world ML applications. All code examples
        are documented in the <code>Projects</code> section.
      </>
    ),
  },
  {
    title: 'Learning in Public',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Track progress through weekly reflections and insights in the blog.
        Documenting the learning journey from fundamentals to mastery,
        one concept at a time.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

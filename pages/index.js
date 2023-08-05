import Link from '@/components/Link';
import { getAllFilesFrontMatter } from '@/lib/mdx';
import BlogPostCard from '@/components/BlogPostCard';
import NowPlaying from '@/components/NowPlaying';

const MAX_DISPLAY = 7;

export async function getStaticProps() {
  const posts = await getAllFilesFrontMatter('blog');

  return { props: { posts } };
}

export default function Home({ posts }) {
  return (
    <>
      {/* <PageSEO
        title={siteMetadata.title + ' - Developer, Writer'}
        description={siteMetadata.description}
      /> */}
      <div>
        <h3 className="font-bold text-xl mt-4 md:text-xl tracking-tight  text-black dark:text-white ">
          Home
        </h3>
        <p className="text-sm leading-7 text-gray-500 my-2 dark:text-gray-400 ">
          I am Kerimcan, a machine learning and AI researcher with a strong passion for education!
          I'm trying to improve myself in the field of Computer Vision.
        </p>
        <div className="flex justify-end text-base font-medium leading-6 mb-6">
          <Link
            href="/about"
            className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400"
            aria-label="blog posts"
          >
            About &rarr;
          </Link>
        </div>
      </div>

      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        <div className="pt-6 pb-4 space-y-2 md:space-y-5">
          <h3 className="font-bold text-xl md:text-xl tracking-tight my-4 text-black dark:text-white">
            Recent Posts
          </h3>
        </div>

        <div className="mb-6">
          {!posts.length && 'No posts found.'}
          {posts.slice(0, MAX_DISPLAY).map((frontMatter, index) => {
            const { slug, title, date } = frontMatter;

            return (
              <BlogPostCard key={slug} date={date} index={index + 1} title={title} slug={slug} />
            );
          })}
        </div>
      </div>

      <div className="flex justify-end text-base font-medium leading-6 mb-6">
        <Link
          href="/blog"
          className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400"
          aria-label="blog posts"
        >
          All Posts &rarr;
        </Link>
      </div>
      <Link href="https://open.spotify.com/user/21l4svwour2yta36ranicusza">
        <NowPlaying />
      </Link>
    </>
  );
}

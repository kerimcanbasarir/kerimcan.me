import { NextSeo } from 'next-seo';

export const MainSEO = () => {
  return (
    <>
      <NextSeo
        title="Kerimcan Başarır"
        description="Python Developer in Turkey"
        openGraph={{
          type: 'website',
          url: 'https://kerimcan.me',
          title: 'Kerimcan Başarır',
          description: 'Python Developer in Turkey',
          images: [{ url: 'https://kerimcan.me/api/og' }],
          siteName: 'Python Developer in Turkey',
        }}
        twitter={{
          handle: '@kerimcanba',
          site: '@kerimcanba',
          cardType: 'summary_large_image',
        }}
        robotsProps={{
          nosnippet: true,
          notranslate: true,
          noimageindex: true,
          noarchive: true,
          maxSnippet: -1,
          maxImagePreview: 'none',
          maxVideoPreview: -1,
        }}
      />
    </>
  );
};

export const BlogSEO = ({ authorDetails, title, summary, date, lastmod, url, tags }) => {
  const publishedAt = new Date(date).toISOString();
  const modifiedAt = new Date(lastmod || date).toISOString();

  return (
    <>
      <NextSeo
        openGraph={{
          title: title,
          description: summary,
          url: url,
          type: 'article',
          article: {
            publishedTime: publishedAt,
            modifiedTime: modifiedAt,
            section: title,

            authors: ['Kerimcan Başarır', 'https://kerimcan.me/'],
            tags: tags,
          },
          images: [
            {
              url: `https://kerimcan.me/api/og?title=${title}`,
              alt: title,
            },
          ],
        }}
      />
    </>
  );
};

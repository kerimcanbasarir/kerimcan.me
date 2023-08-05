import siteMetadata from '@/data/siteMetadata';
import projectsData from '@/data/projectsData';
import Card from '@/components/Card';

export default function Projects() {
  return (
    <>
      {/* <PageSEO title={`Projects - ${siteMetadata.author}`} description={siteMetadata.description} /> */}
      <div className="">
        <div className="pt-6 pb-4 space-y-2 md:space-y-5">
          <h1 className="text-xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-xl sm:leading-10 md:text-xl md:leading-14">
            Projects
          </h1>
          <p className="text-sm leading-7 text-gray-500 dark:text-gray-400">
            Things I Have Made in the Past. Mostly open source
          </p>
        </div>
        <div className="container py-12">
          <div className="flex flex-wrap -m-4">
            {projectsData.map((d) => (
              <Card
                key={d.title}
                title={d.title}
                description={d.description}
                imgSrc={d.imgSrc}
                href={d.href}
                used_tools={d.used_tools}
              />
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

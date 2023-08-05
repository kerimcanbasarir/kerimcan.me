import Image from './Image';
import Link from './Link';

const Card = ({ title, description, imgSrc, href, used_tools }) => (
  <div className="p-2 w-full xs:w-1/2 md:w-1/3">
    <div className="h-full overflow-hidden border-2 border-gray-200 rounded-md border-opacity-60 dark:border-gray-700">
      {imgSrc &&
        (href ? (
          <Link href={href} aria-label={`Link to ${title}`}>
            <Image
              alt={title}
              src={imgSrc}
              className="object-cover object-center lg:h-48 md:h-36"
              width={544}
              height={306}
            />
          </Link>
        ) : (
          <Image
            alt={title}
            src={imgSrc}
            className="object-cover object-center lg:h-48 md:h-36"
            width={544}
            height={306}
          />
        ))}
      <div className="p-3">
        <h2 className="mb-1 text-base font-bold tracking-tight">
          {href ? (
            <Link href={href} aria-label={`Link to ${title}`}>
              {title}
            </Link>
          ) : (
            title
          )}
        </h2>
        <p className="mb-1 prose text-gray-500 max-w-none dark:text-gray-400 text-sm">
          {description}
        </p>
        <p className="mb-1 prose text-gray-500 max-w-none dark:text-gray-400 text-sm">
          <b>Used Tools:</b>
          {used_tools.map((tool, index) => (
            <span key={index}>
              {tool}
              {index !== used_tools.length - 1 ? ', ' : ''}
            </span>
          ))}
        </p>
      </div>
    </div>
  </div>
);

export default Card;

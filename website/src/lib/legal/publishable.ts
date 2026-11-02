// The website has no CI, so the build is the gate: the legal pages call this at
// module level and a Vercel production build fails while draft text remains.
// Preview and local builds still render the draft for review.
export function assertPublishable(
  texts: readonly string[],
  version: string,
  env: string | undefined = process.env.VERCEL_ENV,
): void {
  if (env !== "production") return;
  for (const text of texts) {
    const placeholder = text.match(/\[ASK OWNER[^\]]*\]/);
    if (placeholder) {
      throw new Error(`Legal text still contains ${placeholder[0]}; fill it in before a production build.`);
    }
  }
  if (version.endsWith("-draft")) {
    throw new Error(`Legal text version ${version} is a draft; set the approved version before a production build.`);
  }
}

// The printed TEKCON poster QR points here, so the video link can change
// (Vercel env POSTER_VIDEO_URL + redeploy) without reprinting the poster.
export const dynamic = "force-dynamic";

export function GET(request: Request) {
  // 307, not 308: phones must not cache the destination, it is meant to change.
  return Response.redirect(
    process.env.POSTER_VIDEO_URL || new URL("/", request.url).toString(),
    307,
  );
}

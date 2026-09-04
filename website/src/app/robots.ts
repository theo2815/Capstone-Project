import type { MetadataRoute } from "next";

// Authed surfaces render an empty shell to crawlers (ProtectedRoute returns
// null for anonymous visitors) — keep them out of the index. No sitemap yet:
// /events is force-dynamic, so a sitemap needs a live backend fetch.
export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      allow: "/",
      disallow: [
        "/admin/",
        "/dashboard/",
        "/account",
        "/profile",
        "/orders",
        "/upload/",
        "/verify",
        "/onboarding",
      ],
    },
  };
}

export type Role = "ADMIN" | "PHOTOGRAPHER" | "RUNNER";

export type OAuthProvider = "GOOGLE" | "APPLE";

export interface OAuthIdentity {
  provider: OAuthProvider;
  sub: string;
  email: string;
  name: string;
  avatarUrl?: string;
  // The raw Google ID token, held while the role picker is open so
  // completeOnboarding can re-POST it with the chosen role. Google ID tokens
  // live ~1h; an expired one just bounces the user back to /login.
  idToken: string;
}

export interface User {
  id: string;
  email: string;
  name: string;
  role: Role;
  avatarUrl?: string;
  createdAt: string;
}

export interface AuthResponse {
  accessToken: string;
  refreshToken: string;
  user: User;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  name: string;
  email: string;
  password: string;
  role: Role;
}

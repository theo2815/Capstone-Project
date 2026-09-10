import { useAuthStore } from "@/store/auth-store";

// Capture before starting asynchronous user-scoped work. Identity alone is
// insufficient: logout followed by login to the same account is a new session.
// Also supports the signed-out state while AuthHydrator restores credentials.
export function captureAuthSession(): () => boolean {
  const started = useAuthStore.getState();
  return () => {
    const current = useAuthStore.getState();
    return (
      current.sessionVersion === started.sessionVersion &&
      current.isAuthenticated === started.isAuthenticated &&
      current.user?.id === started.user?.id &&
      current.user?.role === started.user?.role
    );
  };
}

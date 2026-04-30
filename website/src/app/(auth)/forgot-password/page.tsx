import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Forgot Password | QuickPitik",
};

export default function ForgotPasswordPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-gray-900">Reset Password</h1>
      <p className="text-sm text-gray-600">
        Enter your email and we&apos;ll send you a reset link.
      </p>
      {/* ForgotPasswordForm will be implemented here */}
    </div>
  );
}

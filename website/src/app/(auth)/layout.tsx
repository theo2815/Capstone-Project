export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-paper text-ink flex flex-col">
      {children}
    </div>
  );
}

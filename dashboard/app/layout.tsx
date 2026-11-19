import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "RetailFlow",
  description: "Footfall, zones and dwell time from angled CCTV",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <Link href="/" className="brand">RetailFlow</Link>
          <span className="muted">Anonymous footfall analytics · public research videos · times are video time</span>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}

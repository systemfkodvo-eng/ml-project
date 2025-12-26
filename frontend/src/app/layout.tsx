import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "HealthRisk AI - Prédiction des Risques de Santé par IA",
  description: "Plateforme d'intelligence artificielle pour la prédiction des risques de santé. Aidez vos patients avec des diagnostics assistés par IA.",
  keywords: ["IA médicale", "prédiction santé", "machine learning", "diagnostic", "cancer du sein"],
  authors: [{ name: "HealthRisk AI Team" }],
  openGraph: {
    title: "HealthRisk AI - Prédiction des Risques de Santé par IA",
    description: "Plateforme d'intelligence artificielle pour la prédiction des risques de santé.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}

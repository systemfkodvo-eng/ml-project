'use client';

import Link from 'next/link';
import { Brain, Mail, MapPin, Phone, Github, Linkedin, Twitter } from 'lucide-react';

export default function Footer() {
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    product: [
      { label: 'Fonctionnalités', href: '/#features' },
      { label: 'Tarifs', href: '/about' },
      { label: 'Documentation', href: '/faq' },
    ],
    company: [
      { label: 'À propos', href: '/about' },
      { label: 'FAQ', href: '/faq' },
      { label: 'Contact', href: '/about#contact' },
    ],
    legal: [
      { label: 'Confidentialité', href: '/about' },
      { label: 'Conditions d\'utilisation', href: '/about' },
      { label: 'RGPD', href: '/about' },
    ],
  };

  const socialLinks = [
    { icon: Twitter, href: '#', label: 'Twitter' },
    { icon: Linkedin, href: '#', label: 'LinkedIn' },
    { icon: Github, href: '#', label: 'GitHub' },
  ];

  return (
    <footer className="border-t border-slate-800/50 bg-slate-950/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-12">
          {/* Brand Section */}
          <div className="lg:col-span-2">
            <Link href="/" className="flex items-center gap-3 group mb-6">
              <div className="p-2.5 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 shadow-lg shadow-blue-500/20">
                <Brain className="h-7 w-7 text-white" />
              </div>
              <span className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                HealthRisk AI
              </span>
            </Link>
            <p className="text-slate-400 mb-6 max-w-sm leading-relaxed">
              Plateforme d&apos;intelligence artificielle pour la prédiction des risques de santé. 
              Aidez vos patients avec des diagnostics assistés par IA.
            </p>
            <div className="space-y-3">
              <div className="flex items-center gap-3 text-slate-400">
                <Mail className="h-5 w-5 text-blue-400" />
                <span>contact@healthrisk-ai.com</span>
              </div>
              <div className="flex items-center gap-3 text-slate-400">
                <Phone className="h-5 w-5 text-blue-400" />
                <span>+33 1 23 45 67 89</span>
              </div>
              <div className="flex items-center gap-3 text-slate-400">
                <MapPin className="h-5 w-5 text-blue-400" />
                <span>Paris, France</span>
              </div>
            </div>
          </div>

          {/* Product Links */}
          <div>
            <h3 className="text-white font-semibold mb-4">Produit</h3>
            <ul className="space-y-3">
              {footerLinks.product.map((link, index) => (
                <li key={index}>
                  <Link href={link.href} className="text-slate-400 hover:text-blue-400 transition-colors">
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Company Links */}
          <div>
            <h3 className="text-white font-semibold mb-4">Entreprise</h3>
            <ul className="space-y-3">
              {footerLinks.company.map((link, index) => (
                <li key={index}>
                  <Link href={link.href} className="text-slate-400 hover:text-blue-400 transition-colors">
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Legal Links */}
          <div>
            <h3 className="text-white font-semibold mb-4">Légal</h3>
            <ul className="space-y-3">
              {footerLinks.legal.map((link, index) => (
                <li key={index}>
                  <Link href={link.href} className="text-slate-400 hover:text-blue-400 transition-colors">
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-12 pt-8 border-t border-slate-800/50 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-slate-500 text-sm">
            © {currentYear} HealthRisk AI. Tous droits réservés.
          </p>
          <div className="flex items-center gap-4">
            {socialLinks.map((social, index) => {
              const Icon = social.icon;
              return (
                <a
                  key={index}
                  href={social.href}
                  aria-label={social.label}
                  className="p-2 rounded-lg bg-slate-800/50 border border-slate-700/50 text-slate-400 hover:text-blue-400 hover:border-blue-500/30 transition-all"
                >
                  <Icon className="h-5 w-5" />
                </a>
              );
            })}
          </div>
          <p className="text-slate-500 text-xs">
            ⚠️ Outil d&apos;aide à la décision - Ne remplace pas le diagnostic médical
          </p>
        </div>
      </div>
    </footer>
  );
}


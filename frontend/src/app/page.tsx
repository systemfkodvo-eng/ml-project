'use client';

import Link from 'next/link';
import { Brain, Activity, Shield, Sparkles, ArrowRight, Users, FileText, Zap, Heart, TrendingUp, CheckCircle, ChevronDown } from 'lucide-react';
import { useState, useEffect } from 'react';
import PublicNavbar from '@/components/layout/PublicNavbar';
import Footer from '@/components/layout/Footer';
import { publicAPI } from '@/lib/api';

interface PlatformStats {
  model_accuracy: number;
  total_analyses: number;
  total_users: number;
  prediction_time: string;
}

export default function Home() {
  const [scrollY, setScrollY] = useState(0);
  const [platformStats, setPlatformStats] = useState<PlatformStats>({
    model_accuracy: 99.2,
    total_analyses: 0,
    total_users: 0,
    prediction_time: '< 2s'
  });

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await publicAPI.getStats();
        setPlatformStats(data);
      } catch (error) {
        console.error('Error fetching public stats:', error);
      }
    };
    fetchStats();
  }, []);

  const stats = [
    { value: `${platformStats.model_accuracy}%`, label: 'Précision du modèle', icon: TrendingUp },
    { value: platformStats.total_analyses > 0 ? `${platformStats.total_analyses}+` : '0', label: 'Analyses réalisées', icon: FileText },
    { value: platformStats.total_users > 0 ? `${platformStats.total_users}+` : '0', label: 'Praticiens actifs', icon: Users },
    { value: platformStats.prediction_time, label: 'Temps de prédiction', icon: Zap },
  ];

  const features = [
    {
      icon: Brain,
      title: 'IA de Pointe',
      description: 'Modèles de machine learning entraînés sur des données médicales validées pour une précision maximale.',
      gradient: 'from-blue-500 to-cyan-500',
    },
    {
      icon: Shield,
      title: 'Sécurité RGPD',
      description: 'Vos données patient sont pseudonymisées et chiffrées. Conformité totale avec les réglementations.',
      gradient: 'from-emerald-500 to-teal-500',
    },
    {
      icon: Activity,
      title: 'Analyse en Temps Réel',
      description: 'Obtenez des prédictions instantanées pour aider à la prise de décision clinique.',
      gradient: 'from-purple-500 to-pink-500',
    },
    {
      icon: Heart,
      title: 'Validation Humaine',
      description: 'Le praticien reste maître de la décision finale. L\'IA assiste, ne remplace pas.',
      gradient: 'from-rose-500 to-orange-500',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <PublicNavbar />

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
        {/* Animated Background */}
        <div className="absolute inset-0">
          <div
            className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-blue-500/20 rounded-full blur-3xl animate-pulse"
            style={{ transform: `translateY(${scrollY * 0.1}px)` }}
          />
          <div
            className="absolute bottom-1/4 right-1/4 w-[600px] h-[600px] bg-purple-500/20 rounded-full blur-3xl animate-pulse"
            style={{ transform: `translateY(${scrollY * -0.1}px)`, animationDelay: '1s' }}
          />
          <div
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] bg-cyan-500/10 rounded-full blur-3xl animate-pulse"
            style={{ animationDelay: '0.5s' }}
          />
        </div>

        {/* Grid Pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:100px_100px]" />

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 mb-8 animate-fade-in">
            <Sparkles className="h-4 w-4 text-blue-400" />
            <span className="text-sm text-blue-300">Propulsé par l&apos;Intelligence Artificielle</span>
          </div>

          {/* Main Heading */}
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
            Prédiction des Risques
            <br />
            <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent">
              de Santé par IA
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-slate-400 max-w-3xl mx-auto mb-12 leading-relaxed">
            Une plateforme innovante d&apos;aide à la décision clinique utilisant le machine learning
            pour la détection précoce des pathologies.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <Link
              href="/register"
              className="group flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-semibold rounded-2xl shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 transition-all duration-300 text-lg"
            >
              Commencer Gratuitement
              <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="/about"
              className="flex items-center gap-2 px-8 py-4 bg-slate-800/50 hover:bg-slate-700/50 text-white font-semibold rounded-2xl border border-slate-700/50 hover:border-slate-600/50 transition-all duration-300 text-lg"
            >
              En savoir plus
            </Link>
          </div>

          {/* Scroll Indicator */}
          <div className="animate-bounce">
            <ChevronDown className="h-8 w-8 text-slate-500 mx-auto" />
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative py-20 border-y border-slate-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <div key={index} className="text-center group">
                  <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 mb-4 group-hover:scale-110 transition-transform">
                    <Icon className="h-6 w-6 text-blue-400" />
                  </div>
                  <div className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent mb-2">
                    {stat.value}
                  </div>
                  <div className="text-slate-400">{stat.label}</div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4">
              Pourquoi choisir <span className="text-blue-400">HealthRisk AI</span> ?
            </h2>
            <p className="text-xl text-slate-400 max-w-2xl mx-auto">
              Une solution complète pour les professionnels de santé modernes
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className="group p-8 rounded-3xl bg-slate-800/30 border border-slate-700/50 hover:border-slate-600/50 hover:bg-slate-800/50 transition-all duration-300"
                >
                  <div className={`inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br ${feature.gradient} shadow-lg mb-6 group-hover:scale-110 transition-transform`}>
                    <Icon className="h-7 w-7 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-3">{feature.title}</h3>
                  <p className="text-slate-400 leading-relaxed">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-24 border-t border-slate-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4">
              Comment ça <span className="text-cyan-400">fonctionne</span> ?
            </h2>
            <p className="text-xl text-slate-400 max-w-2xl mx-auto">
              Un processus simple en 3 étapes pour des prédictions fiables
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              { step: '01', title: 'Importez les données', desc: 'Téléchargez les documents médicaux ou saisissez les données patient directement.' },
              { step: '02', title: 'Analyse IA', desc: 'Notre modèle ML analyse les données et génère une prédiction de risque.' },
              { step: '03', title: 'Validez et décidez', desc: 'Examinez les résultats, validez les données et prenez votre décision clinique.' },
            ].map((item, index) => (
              <div key={index} className="relative">
                <div className="text-8xl font-bold text-slate-800/50 absolute -top-4 -left-2">{item.step}</div>
                <div className="relative pt-12 pl-4">
                  <h3 className="text-xl font-bold text-white mb-2">{item.title}</h3>
                  <p className="text-slate-400">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="relative rounded-3xl bg-gradient-to-br from-blue-600/20 to-purple-600/20 border border-blue-500/20 p-12 text-center overflow-hidden">
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:50px_50px]" />
            <div className="relative z-10">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 mb-6">
                <CheckCircle className="h-4 w-4 text-emerald-400" />
                <span className="text-sm text-emerald-300">Inscription gratuite</span>
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                Prêt à transformer votre pratique ?
              </h2>
              <p className="text-xl text-slate-300 mb-8 max-w-2xl mx-auto">
                Rejoignez des milliers de praticiens qui utilisent déjà l&apos;IA pour améliorer leurs diagnostics.
              </p>
              <Link
                href="/register"
                className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-semibold rounded-2xl shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 transition-all duration-300 text-lg"
              >
                Créer un compte gratuit
                <ArrowRight className="h-5 w-5" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}

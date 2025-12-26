'use client';

import { Brain, Target, Users, Shield, Award, Heart, Lightbulb, Globe } from 'lucide-react';
import PublicNavbar from '@/components/layout/PublicNavbar';
import Footer from '@/components/layout/Footer';

export default function AboutPage() {
  const values = [
    { icon: Shield, title: 'Sécurité', desc: 'Protection maximale des données patient' },
    { icon: Heart, title: 'Éthique', desc: 'L\'IA au service de l\'humain, jamais l\'inverse' },
    { icon: Lightbulb, title: 'Innovation', desc: 'Recherche continue pour améliorer nos modèles' },
    { icon: Globe, title: 'Accessibilité', desc: 'Rendre l\'IA médicale accessible à tous' },
  ];

  const team = [
    { name: 'Dr. Marie Laurent', role: 'CEO & Co-fondatrice', specialty: 'Médecin & Data Scientist' },
    { name: 'Thomas Dubois', role: 'CTO & Co-fondateur', specialty: 'Expert ML/IA' },
    { name: 'Dr. Sophie Martin', role: 'Directrice Médicale', specialty: 'Oncologue' },
    { name: 'Alexandre Petit', role: 'Lead Developer', specialty: 'Full-Stack Engineer' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <PublicNavbar />

      {/* Hero Section */}
      <section className="pt-32 pb-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-purple-500/10 border border-purple-500/20 mb-8">
            <Target className="h-4 w-4 text-purple-400" />
            <span className="text-sm text-purple-300">Notre Mission</span>
          </div>
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
            Révolutionner le <span className="text-purple-400">diagnostic médical</span>
          </h1>
          <p className="text-xl text-slate-400 max-w-3xl mx-auto">
            HealthRisk AI combine l&apos;expertise médicale et l&apos;intelligence artificielle 
            pour aider les praticiens à détecter les risques de santé plus tôt et plus précisément.
          </p>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-20 border-y border-slate-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
                Notre <span className="text-blue-400">vision</span>
              </h2>
              <p className="text-slate-400 text-lg leading-relaxed mb-6">
                Nous croyons que l&apos;intelligence artificielle peut transformer la médecine 
                en permettant une détection précoce des maladies, tout en respectant l&apos;expertise 
                irremplaçable du médecin.
              </p>
              <p className="text-slate-400 text-lg leading-relaxed">
                Notre plateforme ne remplace pas le diagnostic médical : elle l&apos;enrichit 
                en fournissant des analyses prédictives basées sur des modèles de machine learning 
                entraînés sur des données médicales validées.
              </p>
            </div>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-3xl blur-3xl" />
              <div className="relative bg-slate-800/50 border border-slate-700/50 rounded-3xl p-8">
                <div className="flex items-center gap-4 mb-6">
                  <div className="p-4 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500">
                    <Brain className="h-10 w-10 text-white" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-white">HealthRisk AI</h3>
                    <p className="text-slate-400">Fondée en 2024</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-900/50 rounded-xl p-4 text-center">
                    <div className="text-3xl font-bold text-blue-400">50K+</div>
                    <div className="text-slate-400 text-sm">Analyses</div>
                  </div>
                  <div className="bg-slate-900/50 rounded-xl p-4 text-center">
                    <div className="text-3xl font-bold text-emerald-400">1000+</div>
                    <div className="text-slate-400 text-sm">Praticiens</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Values Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Nos <span className="text-cyan-400">valeurs</span></h2>
          </div>
          <div className="grid md:grid-cols-4 gap-6">
            {values.map((value, index) => {
              const Icon = value.icon;
              return (
                <div key={index} className="text-center p-6 rounded-2xl bg-slate-800/30 border border-slate-700/50 hover:border-slate-600/50 transition-all">
                  <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border border-blue-500/20 mb-4">
                    <Icon className="h-7 w-7 text-blue-400" />
                  </div>
                  <h3 className="text-lg font-bold text-white mb-2">{value.title}</h3>
                  <p className="text-slate-400 text-sm">{value.desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-20 border-t border-slate-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 mb-4">
              <Users className="h-4 w-4 text-emerald-400" />
              <span className="text-sm text-emerald-300">Notre équipe</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-white">Les experts derrière <span className="text-emerald-400">HealthRisk AI</span></h2>
          </div>
          <div className="grid md:grid-cols-4 gap-6">
            {team.map((member, index) => (
              <div key={index} className="text-center p-6 rounded-2xl bg-slate-800/30 border border-slate-700/50 hover:border-slate-600/50 transition-all group">
                <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white text-2xl font-bold">
                  {member.name.split(' ').map(n => n[0]).join('')}
                </div>
                <h3 className="text-lg font-bold text-white mb-1">{member.name}</h3>
                <p className="text-blue-400 text-sm mb-1">{member.role}</p>
                <p className="text-slate-500 text-xs">{member.specialty}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="bg-slate-800/30 border border-slate-700/50 rounded-3xl p-12">
            <Award className="h-12 w-12 text-blue-400 mx-auto mb-6" />
            <h2 className="text-3xl font-bold text-white mb-4">Une question ?</h2>
            <p className="text-slate-400 mb-8">Notre équipe est à votre disposition pour répondre à vos questions.</p>
            <a href="mailto:contact@healthrisk-ai.com" className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-semibold rounded-2xl shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 transition-all">
              Nous contacter
            </a>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}


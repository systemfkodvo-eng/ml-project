'use client';

import { useState } from 'react';
import { HelpCircle, ChevronDown, MessageCircle } from 'lucide-react';
import PublicNavbar from '@/components/layout/PublicNavbar';
import Footer from '@/components/layout/Footer';

interface FAQItem {
  question: string;
  answer: string;
  category: string;
}

const faqData: FAQItem[] = [
  {
    category: 'Général',
    question: 'Qu\'est-ce que HealthRisk AI ?',
    answer: 'HealthRisk AI est une plateforme d\'intelligence artificielle conçue pour aider les professionnels de santé à prédire les risques de pathologies. Elle utilise des modèles de machine learning entraînés sur des données médicales validées pour fournir des analyses prédictives fiables.'
  },
  {
    category: 'Général',
    question: 'L\'IA peut-elle remplacer un médecin ?',
    answer: 'Non, absolument pas. HealthRisk AI est un outil d\'aide à la décision. L\'IA assiste le praticien en fournissant des analyses complémentaires, mais le diagnostic final et la décision thérapeutique restent toujours du ressort du médecin.'
  },
  {
    category: 'Sécurité',
    question: 'Comment mes données sont-elles protégées ?',
    answer: 'Nous appliquons les plus hauts standards de sécurité : chiffrement de bout en bout, pseudonymisation des données patient, hébergement sur des serveurs certifiés HDS (Hébergeur de Données de Santé), et conformité totale avec le RGPD.'
  },
  {
    category: 'Sécurité',
    question: 'Êtes-vous conformes au RGPD ?',
    answer: 'Oui, nous sommes totalement conformes au RGPD. Les données sont traitées en France/Europe, les patients ont un droit d\'accès et de suppression, et nous ne partageons jamais les données avec des tiers sans consentement explicite.'
  },
  {
    category: 'Utilisation',
    question: 'Quels types de prédictions pouvez-vous faire ?',
    answer: 'Actuellement, notre modèle principal se concentre sur la prédiction du cancer du sein à partir de données cytologiques. Nous travaillons sur l\'ajout de nouveaux modèles pour d\'autres pathologies.'
  },
  {
    category: 'Utilisation',
    question: 'Comment importer des données patient ?',
    answer: 'Vous pouvez soit saisir les données manuellement via notre formulaire, soit télécharger des documents médicaux (PDF, images) qui seront analysés automatiquement par notre système d\'extraction de données.'
  },
  {
    category: 'Tarifs',
    question: 'Le service est-il gratuit ?',
    answer: 'Oui, l\'accès à la plateforme est actuellement gratuit pour tous les professionnels de santé. Nous prévoyons d\'introduire des fonctionnalités premium à l\'avenir, mais les fonctionnalités de base resteront gratuites.'
  },
  {
    category: 'Tarifs',
    question: 'Y a-t-il une limite d\'utilisation ?',
    answer: 'Pour le moment, il n\'y a pas de limite stricte. Nous demandons simplement une utilisation raisonnable et éthique de la plateforme, dans le cadre de la pratique médicale.'
  },
];

export default function FAQPage() {
  const [openIndex, setOpenIndex] = useState<number | null>(0);
  const categories = [...new Set(faqData.map(f => f.category))];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <PublicNavbar />

      {/* Hero */}
      <section className="pt-32 pb-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-orange-500/10 border border-orange-500/20 mb-8">
            <HelpCircle className="h-4 w-4 text-orange-400" />
            <span className="text-sm text-orange-300">Centre d&apos;aide</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Questions <span className="text-orange-400">Fréquentes</span>
          </h1>
          <p className="text-xl text-slate-400">
            Trouvez rapidement des réponses à vos questions sur HealthRisk AI
          </p>
        </div>
      </section>

      {/* FAQ Accordion */}
      <section className="py-12 pb-24">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
          {categories.map((category) => (
            <div key={category} className="mb-8">
              <h2 className="text-lg font-semibold text-blue-400 mb-4 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-400"></span>
                {category}
              </h2>
              <div className="space-y-3">
                {faqData.filter(f => f.category === category).map((faq, idx) => {
                  const globalIndex = faqData.indexOf(faq);
                  const isOpen = openIndex === globalIndex;
                  return (
                    <div
                      key={idx}
                      className={`rounded-2xl border transition-all duration-300 ${
                        isOpen 
                          ? 'bg-slate-800/50 border-blue-500/30' 
                          : 'bg-slate-800/30 border-slate-700/50 hover:border-slate-600/50'
                      }`}
                    >
                      <button
                        onClick={() => setOpenIndex(isOpen ? null : globalIndex)}
                        className="w-full px-6 py-5 flex items-center justify-between text-left"
                      >
                        <span className="font-medium text-white pr-4">{faq.question}</span>
                        <ChevronDown className={`h-5 w-5 text-slate-400 transition-transform duration-300 flex-shrink-0 ${isOpen ? 'rotate-180' : ''}`} />
                      </button>
                      <div className={`overflow-hidden transition-all duration-300 ${isOpen ? 'max-h-96' : 'max-h-0'}`}>
                        <div className="px-6 pb-5 text-slate-400 leading-relaxed">
                          {faq.answer}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}

          {/* Contact CTA */}
          <div className="mt-16 text-center p-8 rounded-3xl bg-gradient-to-br from-blue-600/10 to-purple-600/10 border border-blue-500/20">
            <MessageCircle className="h-10 w-10 text-blue-400 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-white mb-2">Vous n&apos;avez pas trouvé votre réponse ?</h3>
            <p className="text-slate-400 mb-6">Notre équipe est là pour vous aider</p>
            <a href="mailto:contact@healthrisk-ai.com" className="inline-flex px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-semibold rounded-xl shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 transition-all">
              Contactez-nous
            </a>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}


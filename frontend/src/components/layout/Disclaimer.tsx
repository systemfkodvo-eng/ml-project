'use client';

import { AlertTriangle, X } from 'lucide-react';
import { useState } from 'react';

export default function Disclaimer() {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  return (
    <div className="relative rounded-xl bg-amber-500/10 border border-amber-500/20 p-4 mb-6 backdrop-blur-sm">
      <button
        onClick={() => setIsVisible(false)}
        className="absolute top-3 right-3 p-1 rounded-lg text-amber-400/60 hover:text-amber-400 hover:bg-amber-500/10 transition-all"
      >
        <X className="h-4 w-4" />
      </button>
      <div className="flex items-start gap-3 pr-8">
        <div className="p-2 rounded-lg bg-amber-500/20">
          <AlertTriangle className="h-5 w-5 text-amber-400" />
        </div>
        <div>
          <p className="text-sm font-semibold text-amber-300">
            Avertissement Important
          </p>
          <p className="text-sm text-amber-200/80 mt-1">
            Cet outil est une <strong className="text-amber-300">aide à la décision</strong>. Il ne remplace en aucun cas
            le diagnostic d&apos;un professionnel de santé qualifié. Toutes les prédictions
            doivent être validées par un praticien agréé.
          </p>
        </div>
      </div>
    </div>
  );
}


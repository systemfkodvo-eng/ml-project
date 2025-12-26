'use client';

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { CheckCircle, ArrowLeft, Sparkles, AlertTriangle, Edit3, FileCheck } from 'lucide-react';
import Navbar from '@/components/layout/Navbar';
import Disclaimer from '@/components/layout/Disclaimer';
import { useAuthStore } from '@/lib/store';
import { extractionAPI, predictionAPI } from '@/lib/api';

interface ExtractedField {
  field_name: string;
  extracted_value: string | number;
  validated_value?: string | number;
  confidence: number;
  entity_type: string;
}

interface Extraction {
  id: string;
  document_id: string;
  patient_id: string;
  extracted_fields: ExtractedField[];
  status: string;
}

export default function ExtractionPage() {
  const router = useRouter();
  const params = useParams();
  const extractionId = params.id as string;
  const { token } = useAuthStore();

  const [extraction, setExtraction] = useState<Extraction | null>(null);
  const [fields, setFields] = useState<ExtractedField[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [validating, setValidating] = useState(false);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    if (!token) {
      router.push('/login');
      return;
    }
    fetchExtraction();
  }, [token, router, extractionId]);

  const fetchExtraction = async () => {
    try {
      const data = await extractionAPI.get(extractionId);
      setExtraction(data);
      setFields(data.extracted_fields.map((f: ExtractedField) => ({
        ...f,
        validated_value: f.validated_value ?? f.extracted_value,
      })));
    } catch (error) {
      console.error('Error fetching extraction:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFieldChange = (index: number, value: string) => {
    const updatedFields = [...fields];
    const numValue = parseFloat(value);
    updatedFields[index].validated_value = isNaN(numValue) ? value : numValue;
    setFields(updatedFields);
  };

  const handleValidate = async () => {
    setValidating(true);
    try {
      await extractionAPI.validate(extractionId, fields);
      await fetchExtraction();
    } catch (error) {
      console.error('Error validating extraction:', error);
    } finally {
      setValidating(false);
    }
  };

  const handlePredict = async () => {
    setPredicting(true);
    try {
      const result = await predictionAPI.run(extractionId);
      router.push(`/analyses/${result.id}`);
    } catch (error) {
      console.error('Error running prediction:', error);
    } finally {
      setPredicting(false);
    }
  };

  if (!token) return null;

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-emerald-400';
    if (confidence >= 0.5) return 'text-amber-400';
    return 'text-rose-400';
  };

  const getConfidenceBg = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-emerald-500/10 border-emerald-500/20';
    if (confidence >= 0.5) return 'bg-amber-500/10 border-amber-500/20';
    return 'bg-rose-500/10 border-rose-500/20';
  };

  const isValidated = extraction?.status === 'validated' || extraction?.status === 'corrected';

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Navbar />

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Disclaimer />

        <button onClick={() => router.back()} className="flex items-center text-slate-400 hover:text-white mb-6 transition-colors">
          <ArrowLeft className="h-4 w-4 mr-2" /> Retour
        </button>

        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-gradient-to-br from-amber-500/20 to-orange-500/20 rounded-xl flex items-center justify-center border border-amber-500/20">
              <Edit3 className="h-5 w-5 text-amber-400" />
            </div>
            <h1 className="text-2xl font-bold text-white">Validation des Données Extraites</h1>
          </div>
          <div className="flex items-center gap-2 text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg px-4 py-3 mt-4">
            <AlertTriangle className="h-5 w-5 flex-shrink-0" />
            <p className="text-sm">
              <strong>Étape obligatoire</strong> : Vérifiez et corrigez les données avant de lancer la prédiction IA
            </p>
          </div>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
          </div>
        ) : !extraction ? (
          <div className="text-center py-12 text-slate-400">Extraction non trouvée</div>
        ) : (
          <>
            <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden">
              <div className="p-5 border-b border-slate-700/50 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FileCheck className="h-5 w-5 text-blue-400" />
                  <div>
                    <h2 className="text-lg font-semibold text-white">Données Extraites</h2>
                    <p className="text-sm text-slate-400">{fields.length} champs détectés par OCR + NER</p>
                  </div>
                </div>
                {isValidated && (
                  <span className="px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full text-sm font-medium flex items-center gap-2">
                    <CheckCircle className="h-4 w-4" /> Validé
                  </span>
                )}
              </div>

              <div className="p-5 space-y-3 max-h-[500px] overflow-y-auto">
                {fields.map((field, index) => (
                  <div key={index} className={`grid grid-cols-12 gap-4 items-center p-4 rounded-xl border ${getConfidenceBg(field.confidence)}`}>
                    <div className="col-span-4">
                      <p className="font-medium text-white">{field.field_name}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <div className={`w-2 h-2 rounded-full ${field.confidence >= 0.8 ? 'bg-emerald-400' : field.confidence >= 0.5 ? 'bg-amber-400' : 'bg-rose-400'}`}></div>
                        <p className={`text-xs ${getConfidenceColor(field.confidence)}`}>
                          Confiance: {Math.round(field.confidence * 100)}%
                        </p>
                      </div>
                    </div>
                    <div className="col-span-3">
                      <p className="text-xs text-slate-500 mb-1">Valeur extraite:</p>
                      <p className="font-mono text-sm text-slate-300 bg-slate-900/50 px-2 py-1 rounded">{String(field.extracted_value)}</p>
                    </div>
                    <div className="col-span-5">
                      <p className="text-xs text-slate-500 mb-1">Valeur validée:</p>
                      <input
                        type="text"
                        value={String(field.validated_value ?? field.extracted_value)}
                        onChange={(e) => handleFieldChange(index, e.target.value)}
                        disabled={isValidated}
                        className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600/50 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 disabled:opacity-50 disabled:cursor-not-allowed"
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="p-5 border-t border-slate-700/50 flex justify-end gap-4">
                {!isValidated ? (
                  <button
                    onClick={handleValidate}
                    disabled={validating}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold rounded-xl shadow-lg shadow-emerald-500/25 disabled:opacity-50 transition-all"
                  >
                    {validating ? (
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    ) : (
                      <CheckCircle className="h-5 w-5" />
                    )}
                    Valider les données
                  </button>
                ) : (
                  <button
                    onClick={handlePredict}
                    disabled={predicting}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-semibold rounded-xl shadow-lg shadow-purple-500/25 disabled:opacity-50 transition-all"
                  >
                    {predicting ? (
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    ) : (
                      <Sparkles className="h-5 w-5" />
                    )}
                    Lancer la prédiction IA
                  </button>
                )}
              </div>
            </div>

            {isValidated && (
              <div className="mt-6 p-5 bg-emerald-500/10 border border-emerald-500/20 rounded-xl flex items-center gap-4">
                <div className="w-12 h-12 bg-emerald-500/20 rounded-xl flex items-center justify-center">
                  <CheckCircle className="h-6 w-6 text-emerald-400" />
                </div>
                <div>
                  <p className="text-emerald-400 font-medium">Données validées avec succès</p>
                  <p className="text-slate-400 text-sm mt-1">
                    Vous pouvez maintenant lancer la prédiction de risque avec le modèle IA entraîné.
                  </p>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}


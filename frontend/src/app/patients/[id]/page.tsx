'use client';

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Upload, FileText, Play, CheckCircle, ArrowLeft, User, Trash2, Eye, Brain, Clock, BarChart3 } from 'lucide-react';
import Navbar from '@/components/layout/Navbar';
import Disclaimer from '@/components/layout/Disclaimer';
import { useAuthStore } from '@/lib/store';
import { patientsAPI, documentsAPI, extractionAPI, predictionAPI } from '@/lib/api';

interface Document {
  id: string;
  filename: string;
  document_type: string;
  status: string;
  created_at: string;
  has_extraction: boolean;
  extraction_id?: string;
}

interface Patient {
  id: string;
  patient_code: string;
  gender?: string;
  notes?: string;
  created_at?: string;
}

interface Analysis {
  id: string;
  risk_level: string;
  risk_percentage: number;
  created_at: string;
}

export default function PatientDetailPage() {
  const router = useRouter();
  const params = useParams();
  const patientId = params.id as string;
  const { token } = useAuthStore();

  const [patient, setPatient] = useState<Patient | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    if (!token) {
      router.push('/login');
      return;
    }
    fetchData();
  }, [token, router, patientId]);

  const fetchData = async () => {
    try {
      const [patientData, docsData, analysesData] = await Promise.all([
        patientsAPI.get(patientId).catch(() => null),
        documentsAPI.listByPatient(patientId).catch(() => []),
        predictionAPI.listAnalyses({ patient_id: patientId }).catch(() => []),
      ]);
      setPatient(patientData);
      setDocuments(Array.isArray(docsData) ? docsData : []);
      setAnalyses(Array.isArray(analysesData) ? analysesData : []);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    try {
      await documentsAPI.upload(patientId, file);
      fetchData();
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setUploading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileUpload(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleProcess = async (documentId: string) => {
    setProcessing(documentId);
    try {
      const extraction = await extractionAPI.process(documentId);
      router.push(`/extraction/${extraction.id}`);
    } catch (error) {
      console.error('Error processing document:', error);
    } finally {
      setProcessing(null);
    }
  };

  const handleDeleteDoc = async (docId: string) => {
    if (confirm('Supprimer ce document ?')) {
      try {
        await documentsAPI.delete(docId);
        fetchData();
      } catch (error) {
        console.error('Error deleting document:', error);
      }
    }
  };

  if (!token) return null;

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'uploaded':
        return <span className="px-3 py-1 bg-slate-500/10 text-slate-400 border border-slate-500/20 rounded-full text-xs font-medium">En attente</span>;
      case 'extracted':
        return <span className="px-3 py-1 bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded-full text-xs font-medium">Extrait</span>;
      case 'validated':
        return <span className="px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full text-xs font-medium">Validé</span>;
      default:
        return <span className="px-3 py-1 bg-slate-500/10 text-slate-400 border border-slate-500/20 rounded-full text-xs">{status}</span>;
    }
  };

  const getRiskBadge = (level: string) => {
    switch (level) {
      case 'low': return <span className="px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full text-xs font-medium">Faible</span>;
      case 'moderate': return <span className="px-3 py-1 bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-full text-xs font-medium">Modéré</span>;
      case 'high': return <span className="px-3 py-1 bg-orange-500/10 text-orange-400 border border-orange-500/20 rounded-full text-xs font-medium">Élevé</span>;
      case 'very_high': return <span className="px-3 py-1 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded-full text-xs font-medium">Très Élevé</span>;
      default: return <span className="px-3 py-1 bg-slate-500/10 text-slate-400 rounded-full text-xs">{level}</span>;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Navbar />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Disclaimer />

        <button onClick={() => router.push('/patients')} className="flex items-center text-slate-400 hover:text-white mb-6 transition-colors">
          <ArrowLeft className="h-4 w-4 mr-2" /> Retour aux patients
        </button>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
          </div>
        ) : !patient ? (
          <div className="text-center py-12 text-slate-400">Patient non trouvé</div>
        ) : (
          <div className="space-y-8">
            {/* Patient Header */}
            <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl border border-slate-700/50 p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-5">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-2xl flex items-center justify-center border border-blue-500/20">
                    <User className="h-8 w-8 text-blue-400" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-white">{patient.patient_code}</h1>
                    <p className="text-slate-400 mt-1">
                      {patient.gender === 'male' ? 'Homme' : patient.gender === 'female' ? 'Femme' : 'Non spécifié'}
                      {patient.created_at && ` • Créé le ${new Date(patient.created_at).toLocaleDateString('fr-FR')}`}
                    </p>
                    {patient.notes && <p className="text-slate-500 mt-2 text-sm">{patient.notes}</p>}
                  </div>
                </div>
                <div className="flex items-center gap-4 text-slate-400">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">{documents.length}</div>
                    <div className="text-xs">Documents</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">{analyses.length}</div>
                    <div className="text-xs">Analyses</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Upload Zone */}
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all ${
                dragActive
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-slate-700 hover:border-slate-600 bg-slate-800/20'
              }`}
            >
              <input
                type="file"
                id="file-upload"
                className="hidden"
                accept=".pdf,.jpg,.jpeg,.png,.csv,.json"
                onChange={handleInputChange}
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <div className="flex flex-col items-center">
                  {uploading ? (
                    <div className="w-12 h-12 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mb-4"></div>
                  ) : (
                    <Upload className={`h-12 w-12 mb-4 ${dragActive ? 'text-blue-400' : 'text-slate-500'}`} />
                  )}
                  <p className="text-white font-medium mb-1">
                    {uploading ? 'Upload en cours...' : 'Glissez un fichier ici ou cliquez pour sélectionner'}
                  </p>
                  <p className="text-slate-500 text-sm">PDF, Images, CSV, JSON (max 50MB)</p>
                </div>
              </label>
            </div>

            {/* Documents */}
            <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden">
              <div className="p-5 border-b border-slate-700/50">
                <div className="flex items-center gap-3">
                  <FileText className="h-5 w-5 text-blue-400" />
                  <h2 className="text-lg font-semibold text-white">Documents</h2>
                </div>
              </div>

              {documents.length === 0 ? (
                <div className="p-12 text-center">
                  <FileText className="h-12 w-12 text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-400">Aucun document uploadé</p>
                </div>
              ) : (
                <div className="divide-y divide-slate-700/50">
                  {documents.map((doc) => (
                    <div key={doc.id} className="p-5 flex items-center justify-between hover:bg-slate-700/20 transition-colors">
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 bg-blue-500/10 rounded-lg flex items-center justify-center">
                          <FileText className="h-5 w-5 text-blue-400" />
                        </div>
                        <div>
                          <p className="font-medium text-white">{doc.filename}</p>
                          <p className="text-sm text-slate-500">
                            {doc.document_type} • {new Date(doc.created_at).toLocaleDateString('fr-FR')}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        {getStatusBadge(doc.status)}

                        {doc.status === 'uploaded' && (
                          <button
                            onClick={() => handleProcess(doc.id)}
                            disabled={processing === doc.id}
                            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white text-sm font-medium rounded-lg disabled:opacity-50 transition-all"
                          >
                            {processing === doc.id ? (
                              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                            ) : (
                              <Play className="h-4 w-4" />
                            )}
                            Extraire (OCR+NER)
                          </button>
                        )}

                        {(doc.status === 'extracted' || doc.status === 'validated') && doc.extraction_id && (
                          <button
                            onClick={() => router.push(`/extraction/${doc.extraction_id}`)}
                            className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg transition-colors"
                          >
                            <Eye className="h-4 w-4" /> Voir extraction
                          </button>
                        )}

                        {doc.status === 'validated' && (
                          <CheckCircle className="h-5 w-5 text-emerald-400" />
                        )}

                        <button
                          onClick={() => handleDeleteDoc(doc.id)}
                          className="p-2 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition-all"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Analyses History */}
            <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden">
              <div className="p-5 border-b border-slate-700/50">
                <div className="flex items-center gap-3">
                  <Brain className="h-5 w-5 text-purple-400" />
                  <h2 className="text-lg font-semibold text-white">Historique des Analyses</h2>
                </div>
              </div>

              {analyses.length === 0 ? (
                <div className="p-12 text-center">
                  <BarChart3 className="h-12 w-12 text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-400">Aucune analyse effectuée</p>
                </div>
              ) : (
                <div className="divide-y divide-slate-700/50">
                  {analyses.map((analysis) => (
                    <div
                      key={analysis.id}
                      onClick={() => router.push(`/analyses/${analysis.id}`)}
                      className="p-5 flex items-center justify-between hover:bg-slate-700/20 transition-colors cursor-pointer"
                    >
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 bg-purple-500/10 rounded-lg flex items-center justify-center">
                          <Brain className="h-5 w-5 text-purple-400" />
                        </div>
                        <div>
                          <p className="font-medium text-white">Analyse de risque</p>
                          <div className="flex items-center gap-2 text-sm text-slate-500">
                            <Clock className="h-3 w-3" />
                            {new Date(analysis.created_at).toLocaleString('fr-FR')}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <div className="text-lg font-bold text-white">{Math.round(analysis.risk_percentage)}%</div>
                          <div className="text-xs text-slate-500">Score de risque</div>
                        </div>
                        {getRiskBadge(analysis.risk_level)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}


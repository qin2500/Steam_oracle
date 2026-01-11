import { Interpretation, SearchStatus } from '@/lib/types';
import { Loader2 } from 'lucide-react';

interface StatusDisplayProps {
    status: SearchStatus;
    interpretation: Interpretation | null;
}

export function StatusDisplay({ status, interpretation }: StatusDisplayProps) {
    if (status === 'idle') return null;

    return (
        <div className="w-full max-w-4xl mx-auto mb-8 animate-in">
            <div className="flex items-center gap-3 mb-4 text-steam-text/80">
                {status === 'interpreting' || status === 'searching' ? (
                    <Loader2 className="w-4 h-4 animate-spin text-steam-light-blue" />
                ) : null}
                <span className="uppercase tracking-widest text-xs font-semibold">
                    {status === 'interpreting' && "Analyzing Request..."}
                    {status === 'searching' && "Finding Matches..."}
                    {status === 'completed' && "Search Results"}
                    {status === 'error' && "Error Encountered"}
                </span>
            </div>

            {interpretation && (
                <div className="flex flex-wrap gap-2">
                    {Object.entries(interpretation).map(([key, value]) => {
                        if (!value || key === 'search_terms') return null;
                        // Clean up key name (e.g., "primary_genre" -> "Primary Genre")
                        const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

                        return (
                            <div
                                key={key}
                                className="px-3 py-1 bg-steam-blue/30 border border-steam-blue rounded text-sm text-steam-light-blue"
                            >
                                <span className="text-steam-text/60 mr-2">{label}:</span>
                                <span className="font-medium">
                                    {Array.isArray(value) ? value.join(", ") : String(value)}
                                </span>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}

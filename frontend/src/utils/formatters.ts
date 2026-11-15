/**
 * Format milliseconds into MM:SS.SS
 */
export function formatTimeMs(ms: number): string {
  const safeMs = Math.max(0, Math.floor(ms));
  const totalSeconds = safeMs / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const hundredths = Math.floor((safeMs % 1000) / 10);

  const mm = minutes.toString().padStart(2, '0');
  const ss = seconds.toString().padStart(2, '0');
  const hh = hundredths.toString().padStart(2, '0');

  return `${mm}:${ss}.${hh}`;
}

/**
 * Human-readable French labels for review and status fields
 */
export function formatStatusLabel(status: string | undefined): {
  label: string;
  variant: 'default' | 'success' | 'warning' | 'danger' | 'info';
} {
  switch (status?.toLowerCase()) {
    case 'approved':
    case 'cleared':
      return { label: 'Validé / Approuvé', variant: 'success' };
    case 'rejected':
      return { label: 'Rejeté', variant: 'danger' };
    case 'pending':
      return { label: 'En attente de revue', variant: 'warning' };
    case 'synthetic_technical_fixture_not_validated_lsf':
      return { label: 'Fixture technique non validée LSF', variant: 'info' };
    default:
      return { label: status || 'Inconnu', variant: 'default' };
  }
}

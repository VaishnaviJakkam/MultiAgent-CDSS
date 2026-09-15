type Props = {
  title: string;
  value: string | number;
  subtitle?: string;
};

export default function MetricCard({
  title,
  value,
  subtitle,
}: Props) {
  return (
    <div className="metric-card">
      <span className="metric-label">{title}</span>
      <h3>{value}</h3>

      {subtitle && (
        <span className="metric-subtitle">
          {subtitle}
        </span>
      )}
    </div>
  );
}
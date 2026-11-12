type Props = {
  level: string;
};

export default function RiskBadge({
  level,
}: Props) {
  const normalized =
    level?.toLowerCase() || "unknown";

  return (
    <span
      className={`risk-badge risk-${normalized}`}
    >
      {level || "UNKNOWN"}
    </span>
  );
}
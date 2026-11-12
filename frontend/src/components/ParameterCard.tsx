type Props = {
  label: string;
  value: string | number;
  unit?: string;
};

export default function ParameterCard({
  label,
  value,
  unit,
}: Props) {
  return (
    <div className="parameter-card">
      <span>{label}</span>

      <div>
        <strong>{value}</strong>

        {unit && (
          <small>{unit}</small>
        )}
      </div>
    </div>
  );
}
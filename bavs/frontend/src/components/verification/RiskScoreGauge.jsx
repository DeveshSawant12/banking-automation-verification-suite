import { RadialBarChart, RadialBar, PolarAngleAxis } from 'recharts'

const RISK_COLOR = { LOW: '#0E9F6E', MEDIUM: '#C27803', HIGH: '#E02424' }

export default function RiskScoreGauge({ score, riskLevel }) {
  const color = RISK_COLOR[riskLevel] || '#6B7280'
  const data = [{ value: score, fill: color }]

  return (
    <div className="flex flex-col items-center gap-2">
      <RadialBarChart
        width={160}
        height={100}
        cx={80}
        cy={90}
        innerRadius={55}
        outerRadius={75}
        startAngle={180}
        endAngle={0}
        data={data}
      >
        <PolarAngleAxis
          type="number"
          domain={[0, 100]}
          angleAxisId={0}
          tick={false}
        />
        <RadialBar
          background={{ fill: '#E5E7EB' }}
          dataKey="value"
          angleAxisId={0}
          data={data}
          cornerRadius={4}
        />
      </RadialBarChart>
      <div className="text-center -mt-6">
        <p className="text-3xl font-bold" style={{ color }}>
          {score}
        </p>
        <p className="text-xs font-semibold tracking-wider uppercase" style={{ color }}>
          {riskLevel} risk
        </p>
      </div>
    </div>
  )
}

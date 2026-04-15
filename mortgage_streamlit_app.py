import React, { useMemo, useState } from "react";
  const [term, setTerm] = useState(300);

  const result = useMemo(() => simulateLoan(loan, rate, term), [loan, rate, term]);

  const totalInterest = result.data.reduce((a, b) => a + b.interest, 0);

  return (
    <div className="h-screen w-screen bg-black text-white grid grid-cols-12">
      
      {/* LEFT CONTROL PANEL */}
      <div className="col-span-3 p-4 space-y-4 bg-neutral-950 border-r border-white/10">
        <h1 className="text-2xl font-light tracking-tight">Finance Engine</h1>

        <Card className="bg-black/40 border-white/10">
          <CardContent className="p-4 space-y-3">
            <div>
              <label>Loan</label>
              <Input value={loan} onChange={(e) => setLoan(+e.target.value)} />
            </div>
            <div>
              <label>Rate %</label>
              <Input value={rate} onChange={(e) => setRate(+e.target.value)} />
            </div>
            <div>
              <label>Term (months)</label>
              <Input value={term} onChange={(e) => setTerm(+e.target.value)} />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-white/10">
          <CardContent className="p-4">
            <div className="text-sm opacity-70">Monthly Payment</div>
            <div className="text-xl font-light">$
              {result.payment.toFixed(2)}
            </div>

            <div className="text-sm opacity-70 mt-4">Total Interest</div>
            <div className="text-xl font-light">$
              {totalInterest.toFixed(0)}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* CENTER WEBGL SCENE */}
      <div className="col-span-6 relative">
        <Canvas camera={{ position: [0, 0, 8] }}>
          <ambientLight intensity={0.3} />
          <Stars radius={50} depth={50} count={5000} factor={4} />
          <FlowField />
          <OrbitControls enableZoom={false} />
        </Canvas>

        <div className="absolute top-6 left-6 text-white/70 text-sm">
          WebGL Financial Space
        </div>
      </div>

      {/* RIGHT PANEL (CHARTS) */}
      <div className="col-span-3 p-4 bg-neutral-950 border-l border-white/10">
        <h2 className="text-lg font-light mb-4">Balance Curve</h2>

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={result.data}>
            <XAxis dataKey="month" hide />
            <YAxis hide />
            <Tooltip />
            <Line type="monotone" dataKey="balance" stroke="#7dd3fc" dot={false} />
          </LineChart>
        </ResponsiveContainer>

        <h2 className="text-lg font-light mt-6 mb-4">Interest Flow</h2>

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={result.data}>
            <XAxis dataKey="month" hide />
            <YAxis hide />
            <Tooltip />
            <Line type="monotone" dataKey="interest" stroke="#a78bfa" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

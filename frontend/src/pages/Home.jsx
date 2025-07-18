import Upload from "../component/Upload"

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-purple-200 to-indigo-100 p-4">
      <div className="w-full max-w-3xl bg-white/80 rounded-2xl shadow-lg p-8 flex flex-row gap-8 items-start">
        <div className="flex-1 flex flex-col items-center">
          <h1 className="font-mono font-extrabold text-4xl text-indigo-700 mb-2 text-center">Jeff's AI Model</h1>
          <h2 className="font-mono font-semibold text-2xl text-purple-700 mb-6 text-center">BG-Noise Cleaner</h2>
          <Upload />
        </div>
        <div className="flex-1 bg-indigo-50/80 rounded-xl p-6 border border-indigo-200 shadow-inner">
          <h3 className="font-mono font-bold text-lg text-indigo-800 mb-2">What do I think of my model?</h3>
          <p className="font-sans text-gray-700 text-base">
            Is the model's performance good? <br /><strong>not quite...</strong><br />
            What about the learning experience? <br /><strong>WORTH IT!</strong><br />
            <br />
            I learned to train a <strong>U-Net model</strong> using <strong>TensorFlow</strong> on 
            a bunch of clean-noisy audio pair from JacobLinCool's VoiceBank-DEMAND
            dataset. 

            <br />
            <br />
            The hardest parts? Learning how to train in a <strong>Linux environment</strong> and  
            <strong> enable GPU access,</strong> as well as <strong>preprocessing and postprocessing </strong>audio files.

            <br />

            <br />
            If you were wondering why I didn't continue training my model:
            
            <br />
            (1) free platforms offered limited memory
            <br />
            (2) didn't want my laptop to burn
            
          </p>
        </div>
      </div>
    </div>
  )
}
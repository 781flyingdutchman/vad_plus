package dev.miracle.vad_plus

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * VAD Configuration matching the C struct
 */
data class VADConfigInternal(
    var positiveSpeechThreshold: Float = 0.5f,
    var negativeSpeechThreshold: Float = 0.35f,
    var preSpeechPadFrames: Int = 3,
    var redemptionFrames: Int = 24,
    var minSpeechFrames: Int = 9,
    var sampleRate: Int = 16000,
    var frameSamples: Int = 512,
    var endSpeechPadFrames: Int = 3,
    var isDebug: Boolean = false
) {
    val contextSize: Int
        get() = if (sampleRate == 16000) 64 else 32
}

/**
 * VAD Event Types matching the C enum
 */
object VADEventType {
    const val INITIALIZED = 0
    const val SPEECH_START = 1
    const val SPEECH_END = 2
    const val FRAME_PROCESSED = 3
    const val REAL_SPEECH_START = 4
    const val MISFIRE = 5
    const val ERROR = 6
    const val STOPPED = 7
}

/**
 * A growable list of primitive floats. Unlike MutableList<Float> it does not
 * box every sample, so buffering audio does not churn the Java heap.
 */
internal class FloatBuilder(initialCapacity: Int = 16000) {
    private var data = FloatArray(initialCapacity)

    var size = 0
        private set

    operator fun get(index: Int): Float {
        if (index >= size) throw IndexOutOfBoundsException("Index $index, size $size")
        return data[index]
    }

    fun isNotEmpty(): Boolean = size > 0

    fun add(source: FloatArray, length: Int) {
        if (size + length > data.size) {
            data = data.copyOf(maxOf(size + length, data.size * 2))
        }
        System.arraycopy(source, 0, data, size, length)
        size += length
    }

    /** Empties the list, keeping its capacity for the next segment. */
    fun clear() {
        size = 0
    }
}

/** A native-order direct buffer, which ONNX Runtime uses without copying. */
private fun directFloats(count: Int): FloatBuffer =
    ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

/**
 * VAD Handle Internal Implementation
 */
class VADHandleInternal {
    // ONNX Runtime
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    
    var config = VADConfigInternal()
        private set
    
    private val hiddenSize = 128
    private val numLayers = 2
    
    // Inference tensors, created once per session and reused for every frame.
    // Each is backed by a direct buffer that ONNX Runtime reads from and
    // writes to in place, so running a frame allocates no tensor memory.
    private var inputData: FloatBuffer? = null // [1, contextSize + frameSamples]
    private var stateData: FloatBuffer? = null // [2, 1, 128], v6 model state
    private var outputData: FloatBuffer? = null // [1, 1], speech probability
    private var stateOutData: FloatBuffer? = null // [2, 1, 128], next state
    private var inputTensors: Map<String, OnnxTensor> = emptyMap()
    private var outputTensors: Map<String, OnnxTensor> = emptyMap()

    // Context buffer for v6
    private var contextBuffer: FloatArray = FloatArray(0)
    
    // Speech detection state
    @Volatile private var _isSpeaking = false
    
    // JNI-compatible getter
    fun isSpeaking(): Boolean = _isSpeaking
    private var speechFrameCount = 0
    private var silenceFrameCount = 0
    private val speechBuffer = FloatBuilder()
    // Ring of the last preSpeechPadFrames frames, oldest at preSpeechStart
    private var preSpeechFrames: Array<FloatArray> = emptyArray()
    private var preSpeechStart = 0
    private var preSpeechCount = 0
    private var hasEmittedRealStart = false
    // speechBuffer size at the last non-silence frame — endSpeechPadFrames
    // of padding is kept after this point when a segment is emitted
    private var samplesAtLastVoice = 0
    
    // Samples received but not yet processed (less than one frame)
    private var pendingFrame: FloatArray = FloatArray(0)
    private var pendingCount = 0

    // Reused by the JNI bridge to pass pushed audio in (see inputArray)
    private var jniInput: FloatArray = FloatArray(0)
    
    // Audio recording
    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    private val isRecording = AtomicBoolean(false)
    
    // Callback - using native pointer for FFI
    private val callbackLock = ReentrantLock()
    private var callbackPtr: Long = 0
    private var userDataPtr: Long = 0
    private var callbackValid = AtomicBoolean(false)
    
    // Last error
    private var _lastError: String = ""
    
    // JNI-compatible getter
    fun getLastError(): String = _lastError
    
    // Stored speech end data
    private var storedSpeechEndPCM16: ShortArray = ShortArray(0)
    
    init {
        resetStates()
    }
    
    fun resetStates() {
        // v6: single state tensor (2, 1, 128) = 256 floats
        stateData?.let { for (i in 0 until it.capacity()) it.put(i, 0f) }

        // Buffers are only reallocated when the config changes their size
        if (contextBuffer.size != config.contextSize) {
        contextBuffer = FloatArray(config.contextSize)
        } else {
            contextBuffer.fill(0f)
        }
        if (pendingFrame.size != config.frameSamples) {
            pendingFrame = FloatArray(config.frameSamples)
        }
        val preSpeechSize = maxOf(0, config.preSpeechPadFrames)
        if (preSpeechFrames.size != preSpeechSize ||
            preSpeechFrames.any { it.size != config.frameSamples }) {
            preSpeechFrames = Array(preSpeechSize) { FloatArray(config.frameSamples) }
        }
        
        _isSpeaking = false
        speechFrameCount = 0
        silenceFrameCount = 0
        speechBuffer.clear()
        preSpeechStart = 0
        preSpeechCount = 0
        hasEmittedRealStart = false
        samplesAtLastVoice = 0
        pendingCount = 0
    }
    
    fun destroy() {
        invalidateCallback()
        stopListening()
        closeTensors()
        ortSession?.close()
        ortSession = null
        ortEnv?.close()
        ortEnv = null
    }
    
    // MARK: - Model Loading
    
    fun initialize(config: VADConfigInternal, modelPath: String?, context: Context): Int {
        this.config = config
        resetStates()
        
        try {
            // Initialize ONNX Runtime
            Log.d(TAG, "Initializing ONNX Runtime environment...")
            ortEnv = OrtEnvironment.getEnvironment()
            Log.d(TAG, "ONNX Runtime environment created successfully")
            
            // Find model path
            val finalModelPath = if (!modelPath.isNullOrEmpty()) {
                if (!File(modelPath).exists()) {
                    _lastError = "Model file not found at provided path: $modelPath"
                    Log.e(TAG, _lastError)
                    return -2
                }
                Log.d(TAG, "Using provided model path: $modelPath")
                modelPath
            } else {
                Log.d(TAG, "Extracting model from assets...")
                extractModelFromAssets(context)
            }
            
            if (finalModelPath == null) {
                _lastError = "ONNX model not found in assets or provided path"
                Log.e(TAG, _lastError)
                return -2
            }
            
            // Verify model file exists and has content
            val modelFile = File(finalModelPath)
            if (!modelFile.exists() || modelFile.length() == 0L) {
                _lastError = "Model file does not exist or is empty: $finalModelPath"
                Log.e(TAG, _lastError)
                return -2
            }
            Log.d(TAG, "Model file verified: ${modelFile.length()} bytes at $finalModelPath")
            
            Log.d(TAG, "Creating ONNX session options...")
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // Use CPU execution provider only - avoid NNAPI which may not support all operators
            // This prevents the "this model don't Support" error from MirrorManager/NNAPI
            
            Log.d(TAG, "Creating ONNX session from model...")
            ortSession = ortEnv!!.createSession(finalModelPath, sessionOptions)
            Log.d(TAG, "ONNX session created successfully")
            createTensors(ortEnv!!)
            
            // Log model info
            val inputNames = ortSession!!.inputNames
            val outputNames = ortSession!!.outputNames
            Log.d(TAG, "Model inputs: $inputNames, outputs: $outputNames")
            
            sendEvent(VADEventType.INITIALIZED)
            return 0
            
        } catch (e: Exception) {
            _lastError = "Initialization failed: ${e.javaClass.simpleName}: ${e.message}"
            Log.e(TAG, "Initialization error: $_lastError", e)
            // Print full stack trace for debugging
            e.printStackTrace()
            return -2
        }
    }
    
    private fun extractModelFromAssets(context: Context): String? {
        val modelNames = listOf("silero_vad_v6.onnx", "silero_vad.onnx")
        
        for (modelName in modelNames) {
            try {
                // Always overwrite the cached copy so a model bundled by a
                // plugin update is never shadowed by a stale extraction.
                val outputFile = File(context.cacheDir, modelName)
                context.assets.open(modelName).use { inputStream ->
                    FileOutputStream(outputFile).use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }

                if (config.isDebug) {
                    Log.d(TAG, "Extracted model to: ${outputFile.absolutePath}")
                }
                return outputFile.absolutePath

            } catch (e: Exception) {
                if (config.isDebug) {
                    Log.d(TAG, "Model $modelName not found in assets: ${e.message}")
                }
                continue
            }
        }
        
        return null
    }
    
    // MARK: - Audio Capture
    
    fun startListening(): Int {
        if (ortSession == null) {
            _lastError = "VAD not initialized"
            return -2
        }
        
        if (isRecording.get()) {
            return 0 // Already recording
        }
        
        try {
            val channelConfig = AudioFormat.CHANNEL_IN_MONO
            val audioFormat = AudioFormat.ENCODING_PCM_16BIT
            val bufferSize = maxOf(
                AudioRecord.getMinBufferSize(config.sampleRate, channelConfig, audioFormat),
                config.frameSamples * 2 * 4 // At least 4 frames worth
            )
            
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                config.sampleRate,
                channelConfig,
                audioFormat,
                bufferSize
            )
            
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                _lastError = "Failed to initialize AudioRecord"
                return -3
            }
            
            audioRecord?.startRecording()
            isRecording.set(true)
            
            recordingThread = Thread {
                val buffer = ShortArray(config.frameSamples)
                val floatData = FloatArray(config.frameSamples)
                
                while (isRecording.get()) {
                    val readResult = audioRecord?.read(buffer, 0, buffer.size) ?: -1
                    
                    if (readResult > 0) {
                        // Check if callback is still valid before processing
                        if (!callbackValid.get()) {
                            continue
                        }
                        
                        // Convert PCM16 to float
                        for (i in 0 until readResult) {
                            floatData[i] = buffer[i].toFloat() / 32768.0f
                        }
                        
                        processAudioData(floatData, readResult)
                    }
                }
            }.apply {
                name = "VadPlusAudioThread"
                start()
            }
            
            if (config.isDebug) {
                Log.d(TAG, "Audio capture started")
            }
            
            return 0
            
        } catch (e: SecurityException) {
            _lastError = "Microphone permission not granted"
            return -4
        } catch (e: Exception) {
            _lastError = e.message ?: "Failed to start audio capture"
            return -5
        }
    }
    
    fun stopListening() {
        isRecording.set(false)
        
        try {
            recordingThread?.join(1000)
        } catch (e: InterruptedException) {
            // Ignore
        }
        recordingThread = null
        
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            // Ignore cleanup errors
        }
        audioRecord = null
        
        resetStates()
        sendEvent(VADEventType.STOPPED)
    }
    
    // MARK: - Callback Management
    
    fun setCallback(callback: Long, userData: Long) {
        callbackLock.withLock {
            callbackPtr = callback
            userDataPtr = userData
            callbackValid.set(callback != 0L)
        }
    }
    
    fun invalidateCallback() {
        callbackLock.withLock {
            callbackValid.set(false)
            callbackPtr = 0
            userDataPtr = 0
        }
    }
    
    // MARK: - Audio Processing
    
    /**
     * Returns an array of at least [size] floats for the JNI bridge to copy
     * pushed audio into before calling [processAudioData], so pushing audio
     * does not allocate a new Java array for every chunk.
     */
    fun inputArray(size: Int): FloatArray {
        if (jniInput.size < size) {
            jniInput = FloatArray(size)
        }
        return jniInput
    }

    fun processAudioData(data: FloatArray) = processAudioData(data, data.size)
        
    fun processAudioData(data: FloatArray, length: Int) {
        val frameSamples = config.frameSamples
        var offset = 0
        while (offset < length) {
            val count = minOf(length - offset, frameSamples - pendingCount)
            System.arraycopy(data, offset, pendingFrame, pendingCount, count)
            pendingCount += count
            offset += count
            if (pendingCount == frameSamples) {
                pendingCount = 0
                // The frame is only read while it is processed; everything
                // that keeps samples (context, speech buffer, pre-speech
                // ring, frame events) copies them.
                processFrame(pendingFrame)
            }
        }
    }
    
    private fun processFrame(frame: FloatArray) {
        try {
            val probability = runInference(frame)
            
            // Send frame processed event
            sendFrameEvent(probability, probability >= config.positiveSpeechThreshold, frame)
            
            processVADLogic(frame, probability)
            
        } catch (e: Exception) {
            _lastError = e.message ?: "Inference error"
            sendErrorEvent(_lastError, -10)
        }
    }
    
    // MARK: - ONNX Inference (v6)
    
    private fun createTensors(env: OrtEnvironment) {
        closeTensors()
        
        val inputSize = config.frameSamples + config.contextSize
        val stateSize = numLayers * hiddenSize
        val stateShape = longArrayOf(numLayers.toLong(), 1, hiddenSize.toLong())
        val input = directFloats(inputSize)
        val state = directFloats(stateSize)
        val output = directFloats(1)
        val stateOut = directFloats(stateSize)

        // Input with context - shape [1, frameSamples + contextSize]
        val inputTensor = OnnxTensor.createTensor(env, input, longArrayOf(1, inputSize.toLong()))
        // Sample rate - shape [1]
        val srTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(longArrayOf(config.sampleRate.toLong())), longArrayOf(1)
        )
        // State - shape [2, 1, 128]
        val stateTensor = OnnxTensor.createTensor(env, state, stateShape)
        // Pinned outputs: the session writes the results into these buffers
        val outputTensor = OnnxTensor.createTensor(env, output, longArrayOf(1, 1))
        val stateOutTensor = OnnxTensor.createTensor(env, stateOut, stateShape)

        inputData = input
        stateData = state
        outputData = output
        stateOutData = stateOut
        inputTensors = mapOf(
            "input" to inputTensor,
            "sr" to srTensor,
            "state" to stateTensor
        )
        outputTensors = mapOf(
            "output" to outputTensor,
            "stateN" to stateOutTensor
        )
    }
        
    private fun closeTensors() {
        inputTensors.values.forEach { it.close() }
        outputTensors.values.forEach { it.close() }
        inputTensors = emptyMap()
        outputTensors = emptyMap()
        inputData = null
        stateData = null
        outputData = null
        stateOutData = null
    }
        
    private fun runInference(frame: FloatArray): Float {
        val session = ortSession ?: throw IllegalStateException("ONNX session not initialized")
        val input = inputData ?: throw IllegalStateException("ONNX tensors not initialized")
        val state = stateData!!
        val output = outputData!!
        val stateOut = stateOutData!!
        
        // Input is the previous frame's context followed by this frame
        input.clear()
        input.put(contextBuffer, 0, config.contextSize)
        input.put(frame, 0, config.frameSamples)
        input.rewind()
        
        // Results land in the pinned output buffers; the Result does not own
        // them, so closing it leaves them open for the next frame.
        session.run(inputTensors, outputTensors).close()
        
        val probability = output.get(0)

        // Update state
        state.clear()
        stateOut.rewind()
        state.put(stateOut)
        state.rewind()
        stateOut.rewind()

        // Update context buffer with the tail of the input
        input.position(input.capacity() - config.contextSize)
        input.get(contextBuffer, 0, config.contextSize)
        input.rewind()
        
        return probability
    }
    
    // MARK: - VAD Logic
    
    private fun processVADLogic(frame: FloatArray, probability: Float) {
        if (!_isSpeaking) {
            if (probability >= config.positiveSpeechThreshold) {
                _isSpeaking = true
                speechFrameCount = 1
                silenceFrameCount = 0
                hasEmittedRealStart = false

                // Prepend the pre-speech ring (the frames before this one),
                // then the triggering frame itself — no duplication.
                for (i in 0 until preSpeechCount) {
                    val preFrame = preSpeechFrames[(preSpeechStart + i) % preSpeechFrames.size]
                    speechBuffer.add(preFrame, preFrame.size)
                }
                speechBuffer.add(frame, config.frameSamples)
                samplesAtLastVoice = speechBuffer.size

                sendEvent(VADEventType.SPEECH_START)
            }
        } else {
            speechBuffer.add(frame, config.frameSamples)
            if (probability >= config.negativeSpeechThreshold) {
                // Not silence — the end pad counts from here
                samplesAtLastVoice = speechBuffer.size
            }

            if (probability >= config.positiveSpeechThreshold) {
                speechFrameCount++
                silenceFrameCount = 0
                
                if (!hasEmittedRealStart && speechFrameCount >= config.minSpeechFrames) {
                    hasEmittedRealStart = true
                    sendEvent(VADEventType.REAL_SPEECH_START)
                }
            } else if (probability < config.negativeSpeechThreshold) {
                silenceFrameCount++
                
                if (silenceFrameCount >= config.redemptionFrames) {
                    if (speechFrameCount >= config.minSpeechFrames) {
                        emitSpeechEnd()
                    } else {
                        sendEvent(VADEventType.MISFIRE)
                    }
                    
                    _isSpeaking = false
                    speechFrameCount = 0
                    silenceFrameCount = 0
                    speechBuffer.clear()
                    hasEmittedRealStart = false
                    samplesAtLastVoice = 0
                }
            }
        }

        // Ring of the frames preceding the current one — updated after the
        // state machine so a new segment gets preSpeechPadFrames of true
        // lead-in without duplicating the triggering frame.
        val ringSize = preSpeechFrames.size
        if (ringSize > 0) {
            if (preSpeechCount < ringSize) {
                System.arraycopy(
                    frame, 0,
                    preSpeechFrames[(preSpeechStart + preSpeechCount) % ringSize], 0,
                    config.frameSamples
                )
                preSpeechCount++
            } else {
                // Full: overwrite the oldest frame, which becomes the newest
                System.arraycopy(frame, 0, preSpeechFrames[preSpeechStart], 0, config.frameSamples)
                preSpeechStart = (preSpeechStart + 1) % ringSize
            }
        }
    }
    
    internal fun emitSpeechEnd() {
        // Keep audio up to the last voiced frame plus endSpeechPadFrames of
        // padding; the rest of the redemption-window silence is trimmed.
        val endPadSamples = maxOf(0, config.endSpeechPadFrames) * config.frameSamples
        val keepSamples = minOf(speechBuffer.size, samplesAtLastVoice + endPadSamples)
        
        // Convert to PCM16
        storedSpeechEndPCM16 = ShortArray(keepSamples) { i ->
            val clamped = speechBuffer[i].coerceIn(-1.0f, 1.0f)
            (clamped * 32767).toInt().toShort()
        }
        
        val durationMs = (keepSamples.toDouble() / config.sampleRate * 1000).toInt()
        
        sendSpeechEndEvent(storedSpeechEndPCM16.size, durationMs)
    }
    
    fun forceEndSpeech() {
        if (_isSpeaking && speechBuffer.isNotEmpty() && speechFrameCount >= config.minSpeechFrames) {
            emitSpeechEnd()
        }
        
        _isSpeaking = false
        speechFrameCount = 0
        silenceFrameCount = 0
        speechBuffer.clear()
        hasEmittedRealStart = false
        samplesAtLastVoice = 0
    }
    
    // MARK: - Event Sending (Native Callbacks)
    
    private fun sendEvent(type: Int) {
        if (!callbackValid.get()) return
        
        callbackLock.withLock {
            if (callbackValid.get() && callbackPtr != 0L) {
                nativeSendEvent(callbackPtr, userDataPtr, type)
            }
        }
    }
    
    private fun sendFrameEvent(probability: Float, isSpeech: Boolean, frame: FloatArray) {
        if (!callbackValid.get()) return
        
        callbackLock.withLock {
            if (callbackValid.get() && callbackPtr != 0L) {
                nativeSendFrameEvent(callbackPtr, userDataPtr, probability, isSpeech, frame, frame.size)
            }
        }
    }
    
    private fun sendSpeechEndEvent(audioLength: Int, durationMs: Int) {
        if (!callbackValid.get()) return
        
        callbackLock.withLock {
            if (callbackValid.get() && callbackPtr != 0L) {
                nativeSendSpeechEndEvent(callbackPtr, userDataPtr, storedSpeechEndPCM16, audioLength, durationMs)
            }
        }
    }
    
    private fun sendErrorEvent(message: String, code: Int) {
        if (!callbackValid.get()) return
        
        callbackLock.withLock {
            if (callbackValid.get() && callbackPtr != 0L) {
                nativeSendErrorEvent(callbackPtr, userDataPtr, message, code)
            }
        }
    }
    
    companion object {
        private const val TAG = "VadPlusFFI"
        
        init {
            System.loadLibrary("vad_plus")
        }
        
        // Native methods for sending events to Dart
        @JvmStatic
        private external fun nativeSendEvent(callbackPtr: Long, userDataPtr: Long, type: Int)
        
        @JvmStatic
        private external fun nativeSendFrameEvent(
            callbackPtr: Long, 
            userDataPtr: Long, 
            probability: Float, 
            isSpeech: Boolean,
            frameData: FloatArray,
            frameLength: Int
        )
        
        @JvmStatic
        private external fun nativeSendSpeechEndEvent(
            callbackPtr: Long, 
            userDataPtr: Long, 
            audioData: ShortArray, 
            audioLength: Int, 
            durationMs: Int
        )
        
        @JvmStatic
        private external fun nativeSendErrorEvent(
            callbackPtr: Long, 
            userDataPtr: Long, 
            message: String, 
            code: Int
        )
    }
}

/**
 * Global handle storage for FFI
 */
object VadPlusHandleManager {
    private val handles = ConcurrentHashMap<Long, VADHandleInternal>()
    private var nextHandleId = 1L
    private val lock = ReentrantLock()
    
    // Application context for asset access
    // Note: Using explicit getter/setter for JNI compatibility
    @Volatile
    @JvmField
    var applicationContext: Context? = null
    
    // Explicit method for JNI access - JNI code looks for this method name
    @JvmStatic
    fun getApplicationContext(): Context? = applicationContext
    
    @JvmStatic
    fun createHandle(): Long {
        lock.withLock {
            val handle = VADHandleInternal()
            val id = nextHandleId++
            handles[id] = handle
            return id
        }
    }
    
    @JvmStatic
    fun getHandle(id: Long): VADHandleInternal? = handles[id]
    
    @JvmStatic
    fun removeHandle(id: Long) {
        lock.withLock {
            handles.remove(id)?.destroy()
        }
    }
}

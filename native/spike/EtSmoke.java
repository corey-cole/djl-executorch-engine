import org.measly.executorch.jni.EtNative;

public final class EtSmoke {
    public static void main(String[] args) {
        String pte = args.length > 0 ? args[0] : "add.pte";
        long h = EtNative.loadModule(pte);
        try {
            float[] out = EtNative.forwardFloat(h, new float[]{2.0f}, new float[]{3.0f});
            float v = out[0];
            System.out.println("RESULT=" + v);
            if (v != 5.0f) {
                throw new AssertionError("expected 5.0 but got " + v);
            }
            System.out.println("JVM<->ExecuTorch desktop bridge OK");
        } finally {
            EtNative.destroy(h);
        }
    }
}
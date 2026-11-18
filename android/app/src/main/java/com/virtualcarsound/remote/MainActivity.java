package com.virtualcarsound.remote;

import android.Manifest;
import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothProfile;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.ParcelUuid;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;

import org.json.JSONObject;

import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Remote control for the Virtual Car Sound engine on the Pi (ble_control.py).
 *
 * Scans for the Pi's Bluetooth LE service (no pairing), connects, and shows: voice picker,
 * volume, idle hum, and switches for effects / virtual gears / small speaker / cruise quiet,
 * plus the live status the Pi sends once a second. Reconnects on its own if the link drops.
 */
public class MainActivity extends Activity {
    static final UUID SERVICE = UUID.fromString("7a6b0001-3c1e-4b7a-9d55-2f1c0e5b8a01");
    static final UUID COMMAND = UUID.fromString("7a6b0002-3c1e-4b7a-9d55-2f1c0e5b8a01");
    static final UUID STATUS = UUID.fromString("7a6b0003-3c1e-4b7a-9d55-2f1c0e5b8a01");
    static final UUID LIST = UUID.fromString("7a6b0004-3c1e-4b7a-9d55-2f1c0e5b8a01");
    static final UUID CCCD = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb");

    final Handler ui = new Handler(Looper.getMainLooper());
    BluetoothAdapter adapter;
    BluetoothLeScanner scanner;
    BluetoothGatt gatt;
    boolean scanning, connected, updatingUi;

    // Android allows one GATT operation at a time: queue them, run the next on completion
    final ArrayDeque<Runnable> ops = new ArrayDeque<>();
    boolean opBusy;

    TextView link, speed, info, volumeLabel, humLabel;
    Spinner voice;
    ArrayAdapter<String> voiceAdapter;
    final List<String> voiceNames = new ArrayList<>();
    final List<String> incoming = new ArrayList<>();
    String currentVoice = "";
    SeekBar volume, hum;
    Switch effects, gears, small, cruise;

    // --- UI -----------------------------------------------------------------------------
    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(20);
        col.setPadding(pad, pad, pad, pad);

        TextView title = text("Virtual Car Sound", 26, true);
        col.addView(title);
        link = text("Starting…", 15, false);
        link.setTextColor(Color.parseColor("#9aa0a6"));
        col.addView(link);

        speed = text("— km/h", 44, true);
        speed.setGravity(Gravity.CENTER_HORIZONTAL);
        speed.setPadding(0, dp(18), 0, 0);
        col.addView(speed);
        info = text("", 15, false);
        info.setGravity(Gravity.CENTER_HORIZONTAL);
        info.setTypeface(Typeface.MONOSPACE);
        col.addView(info);

        col.addView(header("Voice"));
        voice = new Spinner(this);
        voiceAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, voiceNames);
        voice.setAdapter(voiceAdapter);
        voice.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> p, View v, int pos, long id) {
                if (updatingUi || pos < 0 || pos >= voiceNames.size()) return;
                String name = voiceNames.get(pos);
                if (!name.equals(currentVoice)) send("voice " + name);
            }
            @Override public void onNothingSelected(AdapterView<?> p) { }
        });
        col.addView(voice);

        volumeLabel = header("Volume");
        col.addView(volumeLabel);
        volume = slider(v -> send("volume " + (v / 100.0)));
        col.addView(volume);
        humLabel = header("Idle hum when stopped");
        col.addView(humLabel);
        hum = slider(v -> send("set idle_hum " + (v / 100.0)));
        col.addView(hum);

        col.addView(header("Options"));
        effects = toggle(col, "Drive effects (chime, launch, shifts…)", on -> send("set effects " + (on ? 1 : 0)));
        gears = toggle(col, "Virtual gears", on -> send("set virtual_gears " + (on ? 1 : 0)));
        cruise = toggle(col, "Quieter when cruising", on -> send("set cruise_quiet " + (on ? 1 : 0)));
        small = toggle(col, "Small speaker mode", on -> send("set small_speaker " + (on ? 1 : 0)));

        ScrollView scroll = new ScrollView(this);
        scroll.addView(col);
        setContentView(scroll);
        setControlsEnabled(false);
        start();
    }

    interface IntAction { void run(int v); }
    interface BoolAction { void run(boolean v); }

    TextView text(String s, int sp, boolean bold) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(sp);
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    TextView header(String s) {
        TextView t = text(s, 14, true);
        t.setPadding(0, dp(22), 0, dp(6));
        t.setTextColor(Color.parseColor("#8ab4f8"));
        return t;
    }

    SeekBar slider(IntAction onRelease) {
        SeekBar b = new SeekBar(this);
        b.setMax(100);
        b.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar s, int v, boolean user) { }
            @Override public void onStartTrackingTouch(SeekBar s) { }
            @Override public void onStopTrackingTouch(SeekBar s) { if (!updatingUi) onRelease.run(s.getProgress()); }
        });
        return b;
    }

    Switch toggle(LinearLayout col, String label, BoolAction action) {
        Switch s = new Switch(this);
        s.setText(label);
        s.setTextSize(16);
        s.setPadding(0, dp(8), 0, dp(8));
        s.setOnCheckedChangeListener((b, on) -> { if (!updatingUi) action.run(on); });
        col.addView(s, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        return s;
    }

    int dp(int v) { return Math.round(v * getResources().getDisplayMetrics().density); }

    void setLink(String s) { ui.post(() -> link.setText(s)); }

    void setControlsEnabled(boolean on) {
        for (View v : new View[]{voice, volume, hum, effects, gears, cruise, small}) {
            v.setEnabled(on);
            v.setAlpha(on ? 1f : 0.4f);
        }
    }

    // --- permissions & scanning ---------------------------------------------------------
    String[] neededPermissions() {
        if (Build.VERSION.SDK_INT >= 31)
            return new String[]{Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT};
        return new String[]{Manifest.permission.ACCESS_FINE_LOCATION};
    }

    boolean hasPermissions() {
        for (String p : neededPermissions())
            if (checkSelfPermission(p) != PackageManager.PERMISSION_GRANTED) return false;
        return true;
    }

    void start() {
        if (!hasPermissions()) {
            requestPermissions(neededPermissions(), 1);
            return;
        }
        BluetoothManager bm = getSystemService(BluetoothManager.class);
        adapter = bm == null ? null : bm.getAdapter();
        if (adapter == null) { setLink("This phone has no Bluetooth"); return; }
        if (!adapter.isEnabled()) {
            setLink("Turn Bluetooth on, then reopen the app");
            return;
        }
        scan();
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] results) {
        if (hasPermissions()) start();
        else setLink("Bluetooth permission is needed to find the car");
    }

    @SuppressWarnings("MissingPermission")
    void scan() {
        if (scanning || connected) return;
        scanner = adapter.getBluetoothLeScanner();
        if (scanner == null) { retryLater(); return; }
        setLink("Looking for the car…");
        ScanFilter filter = new ScanFilter.Builder().setServiceUuid(new ParcelUuid(SERVICE)).build();
        ScanSettings settings = new ScanSettings.Builder()
                .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build();
        scanning = true;
        scanner.startScan(Collections.singletonList(filter), settings, scanCallback);
        ui.postDelayed(() -> {                       // nothing found: pause, then try again
            if (scanning && !connected) { stopScan(); retryLater(); }
        }, 15000);
    }

    @SuppressWarnings("MissingPermission")
    void stopScan() {
        if (scanning && scanner != null) scanner.stopScan(scanCallback);
        scanning = false;
    }

    void retryLater() {
        setLink("Car not found — retrying…");
        ui.postDelayed(this::scan, 3000);
    }

    final ScanCallback scanCallback = new ScanCallback() {
        @Override
        @SuppressWarnings("MissingPermission")
        public void onScanResult(int type, ScanResult result) {
            if (connected || gatt != null) return;
            stopScan();
            BluetoothDevice d = result.getDevice();
            setLink("Connecting…");
            gatt = d.connectGatt(MainActivity.this, false, gattCallback, BluetoothDevice.TRANSPORT_LE);
        }
    };

    // --- GATT ---------------------------------------------------------------------------
    void queue(Runnable op) {
        synchronized (ops) { ops.add(op); }
        nextOp();
    }

    void nextOp() {
        Runnable op;
        synchronized (ops) {
            if (opBusy || ops.isEmpty()) return;
            opBusy = true;
            op = ops.poll();
        }
        op.run();
    }

    void opDone() {
        synchronized (ops) { opBusy = false; }
        nextOp();
    }

    @SuppressWarnings("MissingPermission")
    void send(String command) {
        if (gatt == null || !connected) return;
        BluetoothGattService s = gatt.getService(SERVICE);
        if (s == null) return;
        BluetoothGattCharacteristic c = s.getCharacteristic(COMMAND);
        byte[] data = command.getBytes(StandardCharsets.UTF_8);
        queue(() -> {
            boolean ok;
            if (Build.VERSION.SDK_INT >= 33) {
                ok = gatt.writeCharacteristic(c, data, BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT)
                        == BluetoothGatt.GATT_SUCCESS;
            } else {
                c.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT);
                c.setValue(data);
                ok = gatt.writeCharacteristic(c);
            }
            if (!ok) opDone();
        });
    }

    @SuppressWarnings("MissingPermission")
    void subscribe(BluetoothGattCharacteristic c) {
        queue(() -> {
            gatt.setCharacteristicNotification(c, true);
            BluetoothGattDescriptor d = c.getDescriptor(CCCD);
            if (d == null) { opDone(); return; }
            boolean ok;
            if (Build.VERSION.SDK_INT >= 33) {
                ok = gatt.writeDescriptor(d, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
                        == BluetoothGatt.GATT_SUCCESS;
            } else {
                d.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                ok = gatt.writeDescriptor(d);
            }
            if (!ok) opDone();
        });
    }

    final BluetoothGattCallback gattCallback = new BluetoothGattCallback() {
        @Override
        @SuppressWarnings("MissingPermission")
        public void onConnectionStateChange(BluetoothGatt g, int status, int newState) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                connected = true;
                setLink("Connected — setting up…");
                g.requestMtu(247);                    // room for status JSON and voice names
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                connected = false;
                synchronized (ops) { ops.clear(); opBusy = false; }
                g.close();
                gatt = null;
                ui.post(() -> setControlsEnabled(false));
                retryLater();
            }
        }

        @Override
        @SuppressWarnings("MissingPermission")
        public void onMtuChanged(BluetoothGatt g, int mtu, int status) {
            g.discoverServices();
        }

        @Override
        public void onServicesDiscovered(BluetoothGatt g, int status) {
            BluetoothGattService s = g.getService(SERVICE);
            if (s == null) { setLink("Car found, but its control service is missing"); return; }
            subscribe(s.getCharacteristic(STATUS));
            subscribe(s.getCharacteristic(LIST));
            send("voices");
            send("status");
            setLink("Connected");
            ui.post(() -> setControlsEnabled(true));
        }

        @Override
        public void onDescriptorWrite(BluetoothGatt g, BluetoothGattDescriptor d, int status) { opDone(); }

        @Override
        public void onCharacteristicWrite(BluetoothGatt g, BluetoothGattCharacteristic c, int status) { opDone(); }

        @Override   // Android 13+
        public void onCharacteristicChanged(BluetoothGatt g, BluetoothGattCharacteristic c, byte[] value) {
            received(c.getUuid(), value);
        }

        @Override   // Android 8-12
        @SuppressWarnings("deprecation")
        public void onCharacteristicChanged(BluetoothGatt g, BluetoothGattCharacteristic c) {
            if (Build.VERSION.SDK_INT < 33) received(c.getUuid(), c.getValue());
        }
    };

    // --- data from the Pi ---------------------------------------------------------------
    void received(UUID uuid, byte[] value) {
        String s = new String(value, StandardCharsets.UTF_8);
        if (LIST.equals(uuid)) {
            if (s.equals("#end")) {
                List<String> names = new ArrayList<>(incoming);
                incoming.clear();
                ui.post(() -> {
                    updatingUi = true;
                    voiceNames.clear();
                    voiceNames.addAll(names);
                    voiceAdapter.notifyDataSetChanged();
                    int i = voiceNames.indexOf(currentVoice);
                    if (i >= 0) voice.setSelection(i);
                    updatingUi = false;
                });
            } else {
                incoming.add(s);
            }
        } else if (STATUS.equals(uuid)) {
            try {
                JSONObject j = new JSONObject(s);
                ui.post(() -> showStatus(j));
            } catch (Exception ignored) { }
        }
    }

    void showStatus(JSONObject j) {
        updatingUi = true;
        try {
            currentVoice = j.optString("v");
            String msg = j.optString("msg");      // e.g. a voice the Pi couldn't load
            link.setText(msg.isEmpty() ? "Connected" : msg);
            link.setTextColor(Color.parseColor(msg.isEmpty() ? "#9aa0a6" : "#f28b82"));
            speed.setText(String.format(java.util.Locale.US, "%.0f km/h", j.optDouble("kmh", 0)));
            String line = j.optString("info");
            info.setText((j.optBoolean("car", true) ? "" : "no car data   ") + line);
            int vol = (int) Math.round(j.optDouble("vol", 1) * 100);
            int ih = (int) Math.round(j.optDouble("ih", 0) * 100);
            if (!volume.isPressed()) volume.setProgress(vol);
            if (!hum.isPressed()) hum.setProgress(ih);
            volumeLabel.setText("Volume  " + vol + " %");
            humLabel.setText("Idle hum when stopped  " + (ih == 0 ? "off (fade out)" : ih + " %"));
            effects.setChecked(j.optBoolean("fx"));
            gears.setChecked(j.optBoolean("vg"));
            cruise.setChecked(j.optBoolean("cq"));
            small.setChecked(j.optBoolean("ss"));
            int i = voiceNames.indexOf(currentVoice);
            if (i >= 0 && voice.getSelectedItemPosition() != i) voice.setSelection(i);
        } finally {
            updatingUi = false;
        }
    }

    @Override
    @SuppressWarnings("MissingPermission")
    protected void onDestroy() {
        super.onDestroy();
        ui.removeCallbacksAndMessages(null);
        stopScan();
        if (gatt != null) { gatt.disconnect(); gatt.close(); }
    }
}

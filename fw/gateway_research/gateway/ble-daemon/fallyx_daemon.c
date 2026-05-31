#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <stdbool.h>
#include <time.h>
#include <signal.h>
#include <pthread.h>
#include <math.h>
#include "btlib.h"
#include "cJSON.h"

volatile sig_atomic_t running = 1;

// Work queue structures for threading
typedef enum {
    WORK_TYPE_DATA = 0,
    WORK_TYPE_CONNECTION = 1,
    WORK_TYPE_DISCONNECTION = 2
} work_type_t;

typedef struct work_item {
    work_type_t type;
    int clientnode;
    unsigned char* data;
    int data_len;
    char mac_str[18];
    struct work_item* next;
} work_item_t;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    work_item_t* head;
    work_item_t* tail;
    int count;
} work_queue_t;

// Global work queue and processing thread
static work_queue_t work_queue = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .condition = PTHREAD_COND_INITIALIZER,
    .head = NULL,
    .tail = NULL,
    .count = 0
};

// Connection management queue and thread
static work_queue_t connection_queue = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .condition = PTHREAD_COND_INITIALIZER,
    .head = NULL,
    .tail = NULL,
    .count = 0
};

static pthread_t processing_thread;
static pthread_t connection_thread;
static pthread_t timeout_monitor_thread;

// A struct to hold the state for one connected sensor
typedef struct {
    char mac_str[18];        // Stores MAC address like "AA:BB:CC:DD:EE:FF\0"
    unsigned char *buffer;   // Dynamically allocated buffer for incoming data
    size_t buffer_size;      // The allocated size of the buffer
    size_t data_len;         // The current length of the data in the buffer
    bool in_use;             // Flag to check if this slot is active
    
    // Reassembly state variables
    bool expecting_fragments;
    unsigned int expected_total_samples;
    unsigned int expected_element_size;
    
    // Timeout tracking
    time_t last_activity;           // Last time we received data
    time_t fragment_start_time;     // When fragmentation began
    bool has_incomplete_data;       // Flag for cleanup decisions
} device_state_t;

#define MAX_CONCURRENT_DEVICES 10
static device_state_t device_pool[MAX_CONCURRENT_DEVICES];
#define MAX_REASSEMBLY_BUFFER_SIZE (64 * 1024)

// Timeout configuration
#define DATA_TIMEOUT_SECONDS 15        // No data for 15 seconds = device timeout
#define FRAGMENT_TIMEOUT_SECONDS 10    // Incomplete packet for 10 seconds = fragment timeout
// Packet structure definitions
typedef struct {
    unsigned int version;
    unsigned int id1;
    unsigned int id2;
    unsigned int id3;
    unsigned int tick;
    unsigned int config_id;
    unsigned int message_id;
    unsigned char packet_type;
} FallyxPacketHeader;

typedef struct {
    unsigned char packet_format;
    unsigned short num_samples;
    unsigned char element_size;
} IMUPayloadHeader;

typedef struct {
    float accel_x;
    float accel_y;
    float accel_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
    unsigned char aux_data[8];
    unsigned int timestamp;
} IMUDataSample;

typedef struct {
    unsigned int timestamp;     // when this measurement was taken
    float voltage;             // in mV
    float current;             // in mA
    float avg_power;           // in mW
    float remaining;           // in mAh
    float temperature;         // in celsius
    float pressure;            // in hPa
    unsigned char fsr1;        // analog (255 = default)
    unsigned char fsr2;        // analog (255 = default)
    unsigned char button;      // 0 or 255
    unsigned char AccelRate;   // NEW: Accelerometer rate
    unsigned char GyroRate;    // NEW: Gyroscope rate
    unsigned char AccelDownsample; // NEW: Accelerometer downsample
    unsigned char GyroDownsample;  // NEW: Gyroscope downsample
} TempBattGpioPayload;

typedef struct {
    unsigned int timestamp;  // when this measurement was taken
    float pressure;
} PressurePayload;

// --- Forward Declarations ---
static char* process_and_serialize_packet(const unsigned char* data, size_t len, const char* mac_str);
static char* process_incomplete_packet(const unsigned char* data, size_t len, const char* mac_str, const char* reason);
void send_json_to_python(const char* json_string);
static void* processing_thread_func(void* arg);
static void* connection_thread_func(void* arg);
static void* timeout_monitor_thread_func(void* arg);
static bool is_valid_pressure(float pressure, unsigned int timestamp);
static work_item_t* create_connection_work_item(int clientnode, const char* mac_str, work_type_t type);
static void queue_connection_item(work_item_t* item);
static work_item_t* dequeue_connection_item(void);
static void cleanup_connection_queue(void);

// --- Globals & Constants ---
int py_socket_fd = -1;
const char* SOCKET_PATH = "/tmp/fallyx_ble.sock";
int data_tx_ctic_index = -1;  // We listen on DataTx from sensor's perspective

// --- Socket Communication Code ---
void connect_to_python_socket() {
    struct sockaddr_un addr;
    if ((py_socket_fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
        perror("socket error");
        exit(-1);
    }
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    printf("Connecting to Python socket at %s...\n", SOCKET_PATH);
    while (connect(py_socket_fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        printf("Failed to connect to Python socket. Retrying in 2 seconds...\n");
        sleep(2);
    }
    printf("Connected to Python socket successfully.\n");
}
static void release_device_state(const char* mac_str) {
    if (!mac_str) return;

    for (int i = 0; i < MAX_CONCURRENT_DEVICES; i++) {
        if (device_pool[i].in_use && strcmp(device_pool[i].mac_str, mac_str) == 0) {
            printf("Releasing state for disconnected device: %s\n", mac_str);
            free(device_pool[i].buffer); // IMPORTANT: Free the allocated buffer
            device_pool[i].buffer = NULL;
            device_pool[i].in_use = false;
            // Optionally, zero out the whole struct for good hygiene
            memset(&device_pool[i], 0, sizeof(device_state_t));
            return; // Found and released
        }
    }
    printf("Warning: Could not find state to release for disconnected device %s\n", mac_str);
}
// Finds an existing device state by MAC or initializes a new one.
// Returns NULL if the pool is full.
static device_state_t* get_or_create_device_state(const char* mac_str) {
    // 1. Search for an existing device with the same MAC address.
    for (int i = 0; i < MAX_CONCURRENT_DEVICES; i++) {
        if (device_pool[i].in_use && strcmp(device_pool[i].mac_str, mac_str) == 0) {
            return &device_pool[i];
        }
    }

    // 2. If not found, find a free slot to create a new one.
    for (int i = 0; i < MAX_CONCURRENT_DEVICES; i++) {
        if (!device_pool[i].in_use) {
            device_state_t* new_state = &device_pool[i];
            strncpy(new_state->mac_str, mac_str, sizeof(new_state->mac_str) - 1);
            new_state->mac_str[sizeof(new_state->mac_str) - 1] = '\0';
            
            // Initial buffer allocation (e.g., 4KB)
            new_state->buffer_size = 4096;
            new_state->buffer = (unsigned char*)malloc(new_state->buffer_size);
            if (new_state->buffer == NULL) {
                perror("Failed to allocate buffer for new device");
                return NULL; // Allocation failed
            }
            
            new_state->data_len = 0;
            new_state->in_use = true;
            new_state->expecting_fragments = false;
            new_state->expected_total_samples = 0;
            new_state->expected_element_size = 0;
            new_state->last_activity = time(NULL);
            new_state->fragment_start_time = 0;
            new_state->has_incomplete_data = false;
            printf("Created new state for device: %s\n", mac_str);
            return new_state;
        }
    }

    printf("Error: Device pool is full. Cannot add new device %s.\n", mac_str);
    return NULL; // No free slots
}

// Initialize the device pool
static void init_device_pool() {
    for (int i = 0; i < MAX_CONCURRENT_DEVICES; i++) {
        device_pool[i].in_use = false;
        device_pool[i].buffer = NULL;
        device_pool[i].data_len = 0;
        device_pool[i].buffer_size = 0;
        device_pool[i].expecting_fragments = false;
        device_pool[i].expected_total_samples = 0;
        device_pool[i].expected_element_size = 0;
        device_pool[i].last_activity = 0;
        device_pool[i].fragment_start_time = 0;
        device_pool[i].has_incomplete_data = false;
        memset(device_pool[i].mac_str, 0, sizeof(device_pool[i].mac_str));
    }
}
// --- Work Queue Management Functions ---
static work_item_t* create_work_item(int clientnode, const unsigned char* data, int data_len, const char* mac_str) {
    work_item_t* item = (work_item_t*)malloc(sizeof(work_item_t));
    if (!item) {
        printf("Error: Failed to allocate work item\n");
        return NULL;
    }
    
    item->clientnode = clientnode;
    item->data_len = data_len;
    item->data = (unsigned char*)malloc(data_len);
    if (!item->data) {
        printf("Error: Failed to allocate work item data\n");
        free(item);
        return NULL;
    }
    
    memcpy(item->data, data, data_len);
    strncpy(item->mac_str, mac_str, sizeof(item->mac_str) - 1);
    item->mac_str[sizeof(item->mac_str) - 1] = '\0';
    item->next = NULL;
    
    return item;
}

static work_item_t* create_connection_work_item(int clientnode, const char* mac_str, work_type_t type) {
    work_item_t* item = (work_item_t*)malloc(sizeof(work_item_t));
    if (!item) {
        printf("Error: Failed to allocate connection work item\n");
        return NULL;
    }
    
    item->type = type;
    item->clientnode = clientnode;
    item->data = NULL;
    item->data_len = 0;
    strncpy(item->mac_str, mac_str, sizeof(item->mac_str) - 1);
    item->mac_str[sizeof(item->mac_str) - 1] = '\0';
    item->next = NULL;
    
    return item;
}

static void free_work_item(work_item_t* item) {
    if (item) {
        if (item->data) {
            free(item->data);
        }
        free(item);
    }
}

static void queue_work_item(work_item_t* item) {
    if (!item) return;
    
    pthread_mutex_lock(&work_queue.mutex);
    
    if (work_queue.tail) {
        work_queue.tail->next = item;
    } else {
        work_queue.head = item;
    }
    work_queue.tail = item;
    work_queue.count++;
    
    pthread_cond_signal(&work_queue.condition);
    pthread_mutex_unlock(&work_queue.mutex);
    
    printf("Queued work item (queue size: %d)\n", work_queue.count);
}

static work_item_t* dequeue_work_item() {
    pthread_mutex_lock(&work_queue.mutex);
    
    while (work_queue.head == NULL && running) {
        pthread_cond_wait(&work_queue.condition, &work_queue.mutex);
    }
    
    work_item_t* item = work_queue.head;
    if (item) {
        work_queue.head = item->next;
        if (work_queue.head == NULL) {
            work_queue.tail = NULL;
        }
        work_queue.count--;
        item->next = NULL;
    }
    
    pthread_mutex_unlock(&work_queue.mutex);
    return item;
}

static void queue_connection_item(work_item_t* item) {
    if (!item) return;
    
    pthread_mutex_lock(&connection_queue.mutex);
    
    if (connection_queue.tail) {
        connection_queue.tail->next = item;
    } else {
        connection_queue.head = item;
    }
    connection_queue.tail = item;
    connection_queue.count++;
    
    pthread_cond_signal(&connection_queue.condition);
    pthread_mutex_unlock(&connection_queue.mutex);
    
    printf("Queued connection item (connection queue size: %d)\n", connection_queue.count);
}

static work_item_t* dequeue_connection_item() {
    pthread_mutex_lock(&connection_queue.mutex);
    
    while (connection_queue.head == NULL && running) {
        pthread_cond_wait(&connection_queue.condition, &connection_queue.mutex);
    }
    
    work_item_t* item = connection_queue.head;
    if (item) {
        connection_queue.head = item->next;
        if (connection_queue.head == NULL) {
            connection_queue.tail = NULL;
        }
        connection_queue.count--;
        item->next = NULL;
    }
    
    pthread_mutex_unlock(&connection_queue.mutex);
    return item;
}

static void cleanup_work_queue() {
    pthread_mutex_lock(&work_queue.mutex);
    
    work_item_t* current = work_queue.head;
    while (current) {
        work_item_t* next = current->next;
        free_work_item(current);
        current = next;
    }
    
    work_queue.head = NULL;
    work_queue.tail = NULL;
    work_queue.count = 0;
    
    pthread_mutex_unlock(&work_queue.mutex);
}

static void cleanup_connection_queue() {
    pthread_mutex_lock(&connection_queue.mutex);
    
    work_item_t* current = connection_queue.head;
    while (current) {
        work_item_t* next = current->next;
        free_work_item(current);
        current = next;
    }
    
    connection_queue.head = NULL;
    connection_queue.tail = NULL;
    connection_queue.count = 0;
    
    pthread_mutex_unlock(&connection_queue.mutex);
}
// --- Processing Thread Function ---
// This function runs in a separate thread and handles all the heavy packet processing
// that was previously done in the callback handler, eliminating the bottleneck
static void* processing_thread_func(void* arg) {
    printf("Processing thread started\n");
    
    while (running) {
        work_item_t* item = dequeue_work_item();
        if (!item) {
            continue; // Shutdown signal received
        }
        
        printf("Processing work item for device %s (%d bytes)\n", item->mac_str, item->data_len);
        
        // Get or create device state (thread-safe since each device has its own state)
        device_state_t* state = get_or_create_device_state(item->mac_str);
        if (!state) {
            printf("Error: Could not get device state for MAC %s\n", item->mac_str);
            free_work_item(item);
            continue;
        }
        
        // Update activity timestamp
        state->last_activity = time(NULL);
        state->has_incomplete_data = (state->data_len > 0);
        
        // Handle packet reassembly for fragmented data - moved from callback
        if (item->data_len >= 4 && item->data[0] == 'C' && item->data[1] == 'M' && 
            item->data[2] == 'D' && item->data[3] == ':') {
            // This is the start of a new packet - reset reassembly buffer
            printf("=== NEW PACKET START for %s ===\n", item->mac_str);
            state->data_len = 0;
            state->expecting_fragments = false;
            
            // Check if we need to resize the buffer
            if (item->data_len > state->buffer_size) {
                size_t new_size = item->data_len * 2;  // Give some extra space
                unsigned char* new_buffer = (unsigned char*)realloc(state->buffer, new_size);
                if (!new_buffer) {
                    printf("Error: Failed to resize buffer for device %s\n", item->mac_str);
                    free_work_item(item);
                    continue;
                }
                state->buffer = new_buffer;
                state->buffer_size = new_size;
                printf("Resized buffer for device %s to %zu bytes\n", item->mac_str, new_size);
            }
            
            // Copy this fragment to reassembly buffer
            memcpy(state->buffer, item->data, item->data_len);
            state->data_len = item->data_len;
            
            // If this fragment has enough data to read the header, extract expected size
            if (item->data_len >= 33) { // CMD:(4) + header(28) + packet_type(1)
                unsigned char packet_type = item->data[32]; // packet_type is at offset 32
                int expected_total_size;
                
                if (packet_type == 0xA1) {
                    // IMU packet - ORIGINAL FORMAT
                    if (item->data_len >= 37) { // Need full payload header
                        unsigned short numSamples = *((unsigned short*)&item->data[34]);
                        unsigned char elementSize = item->data[36];
                        printf("DEBUG, packet_type 0x%02X: numSamples = %u, elementSize = %u\n", packet_type, numSamples, elementSize);
                        state->expected_total_samples = numSamples;
                        state->expected_element_size = elementSize;
                        expected_total_size = 37 + (numSamples * elementSize);
                        printf("Expected total packet size: %d bytes (%u samples × %u bytes + 37 header) [Type 0x%02X]\n",
                               expected_total_size, numSamples, elementSize, packet_type);
                    } else {
                        // Not enough data yet for full header
                        state->expecting_fragments = true;
                        printf("Packet type 0x%02X header incomplete, waiting for more data\n", packet_type);
                        free_work_item(item);
                        continue;
                    }
                } else if (packet_type == 0x45) {
                    // Pressure packet
                    if (item->data_len >= 37) { // Need full payload header
                        unsigned char packetFormat = item->data[33];
                        unsigned short numSamples = *((unsigned short*)&item->data[34]);
                        unsigned char sampleSize = item->data[36];
                        printf("DEBUG, packet_type 0x%02X: packetFormat = 0x%02X, numSamples = %u, sampleSize = %u\n",
                               packet_type, packetFormat, numSamples, sampleSize);
                        state->expected_total_samples = numSamples;
                        state->expected_element_size = sampleSize;
                        expected_total_size = 37 + (numSamples * sampleSize);
                        printf("Expected total packet size: %d bytes (%u samples × %u bytes + 37 header) [Type 0x%02X]\n",
                               expected_total_size, numSamples, sampleSize, packet_type);
                    } else {
                        // Not enough data yet for full header
                        state->expecting_fragments = true;
                        printf("Packet type 0x%02X header incomplete, waiting for more data\n", packet_type);
                        free_work_item(item);
                        continue;
                    }
                } else if (packet_type == 0x41) {
                    // Temp/Batt/GPIO packet - fixed payload size
                    expected_total_size = 4 + 28 + 36; // CMD: + header + payload(36 bytes including packet_format)
                    state->expected_total_samples = 1; // Not used for fixed-size packets
                    state->expected_element_size = 36; // Total payload size including packet_format
                } else {
                    // Unknown packet type - assume minimal size
                    expected_total_size = 4 + 28 + 8; // CMD: + header + minimal payload
                    state->expected_total_samples = 1;
                    state->expected_element_size = 8;
                    printf("Expected total packet size: %d bytes [UNKNOWN TYPE 0x%02X]\n", expected_total_size, packet_type);
                }
                
                if (item->data_len < expected_total_size) {
                    state->expecting_fragments = true;
                    state->fragment_start_time = time(NULL);  // Track when fragmentation started
                    printf("Expecting more fragments (have %d of %d bytes)\n", item->data_len, expected_total_size);
                } else {
                    // Complete packet received in one fragment
                    printf("=== COMPLETE PACKET RECEIVED in one fragment ===\n");
                    char* json_string = process_and_serialize_packet(state->buffer, state->data_len, item->mac_str);
                    if (json_string) {
                        send_json_to_python(json_string);
                        free(json_string);
                    } else {
                        printf("Error: Failed to process packet for device %s\n", item->mac_str);
                    }
                    state->data_len = 0;  // Reset for next packet
                }
            }
        } else if (state->expecting_fragments && state->data_len > 0) {
            // This is a continuation fragment - handle reassembly
            printf("=== FRAGMENT RECEIVED for %s ===\n", item->mac_str);
            printf("Adding %d bytes to existing %zu bytes\n", item->data_len, state->data_len);
            
            // Check if the buffer is overgrown and needs to be abandoned
            if (state->data_len + item->data_len > MAX_REASSEMBLY_BUFFER_SIZE) {
                printf("Error: Packet for device %s exceeds max size. Discarding.\n", item->mac_str);
                state->data_len = 0;
                state->expecting_fragments = false;
                free_work_item(item);
                continue;
            }
            
            // Check if we need to resize the buffer
            if (state->data_len + item->data_len > state->buffer_size) {
                size_t new_size = state->buffer_size * 2;
                while (new_size < state->data_len + item->data_len) {
                    new_size *= 2;
                }
                
                unsigned char* new_buffer = (unsigned char*)realloc(state->buffer, new_size);
                if (!new_buffer) {
                    printf("Error: Failed to resize buffer for device %s\n", item->mac_str);
                    free_work_item(item);
                    continue;
                }
                
                state->buffer = new_buffer;
                state->buffer_size = new_size;
                printf("Resized buffer for device %s to %zu bytes\n", item->mac_str, new_size);
            }
            
            // Append this fragment to reassembly buffer
            memcpy(state->buffer + state->data_len, item->data, item->data_len);
            state->data_len += item->data_len;
            
            // Check if we now have a complete packet
            int expected_total_size;
            if (state->data_len >= 33) {
                unsigned char packet_type = state->buffer[32];
                if (packet_type == 0xA1) {
                    expected_total_size = 37 + (state->expected_total_samples * state->expected_element_size);
                } else if (packet_type == 0x45) {
                    expected_total_size = 37 + (state->expected_total_samples * state->expected_element_size);
                } else if (packet_type == 0x41) {
                    expected_total_size = 4 + 28 + 36;
                } else {
                    expected_total_size = 4 + 28 + 8;
                }
            } else {
                expected_total_size = 37 + (state->expected_total_samples * state->expected_element_size);
            }
            
            printf("Reassembly buffer now has %zu bytes (need %d)\n", state->data_len, expected_total_size);
            
            if (state->data_len >= expected_total_size) {
                printf("=== COMPLETE PACKET ASSEMBLED for %s ===\n", item->mac_str);
                state->expecting_fragments = false;
                state->fragment_start_time = 0;  // Reset fragment timing
                
                // Process the complete packet (only the expected size)
                char* json_string = process_and_serialize_packet(state->buffer, expected_total_size, item->mac_str);
                if (json_string) {
                    send_json_to_python(json_string);
                    free(json_string);
                } else {
                    printf("Error: Failed to process packet for device %s\n", item->mac_str);
                }
                
                // Handle any extra bytes that might be the start of the next packet
                int extra_bytes = state->data_len - expected_total_size;
                if (extra_bytes > 0) {
                    printf("Found %d extra bytes after packet completion - moving to start of buffer\n", extra_bytes);
                    unsigned char* extra_data = state->buffer + expected_total_size;
                    memmove(state->buffer, extra_data, extra_bytes);
                    state->data_len = extra_bytes;
                    
                    // If extra bytes start with "CMD:", we have the start of the next packet
                    if (extra_bytes >= 4 && state->buffer[0] == 'C' && state->buffer[1] == 'M' &&
                        state->buffer[2] == 'D' && state->buffer[3] == ':') {
                        printf("Extra bytes contain start of next packet - will process in next iteration\n");
                        // The next work item will handle this as a new packet start
                    } else {
                        printf("Extra bytes don't start with CMD: - discarding %d bytes\n", extra_bytes);
                        state->data_len = 0;
                    }
                } else {
                    state->data_len = 0;  // Reset buffer for next packet
                }
            } else {
                printf("Still waiting for more fragments for %s\n", item->mac_str);
            }
        } else {
            // Unexpected data - not a CMD: start and not expecting fragments
            printf("Warning: Received unexpected data for device %s (not CMD: start and not expecting fragments)\n", item->mac_str);
            // Reset state to be safe
            state->data_len = 0;
            state->expecting_fragments = false;
        }
        
        free_work_item(item);
    }
    
    printf("Processing thread shutting down\n");
    return NULL;
}

// --- Timeout Monitoring Thread Function ---
// This function runs in a separate thread and monitors device timeouts
static void* timeout_monitor_thread_func(void* arg) {
    printf("Timeout monitoring thread started\n");
    
    while (running) {
        time_t now = time(NULL);
        
        for (int i = 0; i < MAX_CONCURRENT_DEVICES; i++) {
            if (!device_pool[i].in_use) continue;
            
            device_state_t* state = &device_pool[i];
            
            // Check for fragment timeout
            if (state->expecting_fragments && state->fragment_start_time > 0 &&
                (now - state->fragment_start_time) > FRAGMENT_TIMEOUT_SECONDS) {
                
                printf("TIMEOUT: Device %s fragment timeout (%ld seconds)\n",
                       state->mac_str, now - state->fragment_start_time);
                
                // Salvage incomplete data
                if (state->data_len > 0) {
                    char* json_string = process_incomplete_packet(
                        state->buffer, state->data_len,
                        state->mac_str, "fragment_timeout");
                    if (json_string) {
                        send_json_to_python(json_string);
                        free(json_string);
                        printf("Successfully sent incomplete fragment data for device %s\n", state->mac_str);
                    } else {
                        printf("Could not process incomplete fragment data for device %s\n", state->mac_str);
                    }
                }
                
                // Reset fragment state but keep device active
                state->data_len = 0;
                state->expecting_fragments = false;
                state->fragment_start_time = 0;
                state->has_incomplete_data = false;
                printf("Reset fragment state for device %s\n", state->mac_str);
            }
            
            // Check for complete device timeout
            else if (state->last_activity > 0 &&
                     (now - state->last_activity) > DATA_TIMEOUT_SECONDS) {
                
                printf("TIMEOUT: Device %s data timeout (%ld seconds)\n",
                       state->mac_str, now - state->last_activity);
                
                // Salvage any remaining data
                if (state->data_len > 0) {
                    char* json_string = process_incomplete_packet(
                        state->buffer, state->data_len,
                        state->mac_str, "device_timeout");
                    if (json_string) {
                        send_json_to_python(json_string);
                        free(json_string);
                        printf("Successfully sent incomplete timeout data for device %s\n", state->mac_str);
                    } else {
                        printf("Could not process incomplete timeout data for device %s\n", state->mac_str);
                    }
                }
                
                // Release device state (simulate disconnect)
                printf("Releasing timed-out device: %s\n", state->mac_str);
                release_device_state(state->mac_str);
            }
        }
        
        sleep(10); // Check every 10 seconds
    }
    
    printf("Timeout monitoring thread shutting down\n");
    return NULL;
}

// --- Connection Management Thread Function ---
// This function runs in a separate thread and handles connection/disconnection setup
// to prevent blocking the BLE callback during connection events
static void* connection_thread_func(void* arg) {
    printf("Connection management thread started\n");
    
    while (running) {
        work_item_t* item = dequeue_connection_item();
        if (!item) {
            continue; // Shutdown signal received
        }
        
        printf("Processing connection item for device %s (type: %d)\n", item->mac_str, item->type);
        
        if (item->type == WORK_TYPE_CONNECTION) {
            // Handle connection setup - moved from callback to prevent blocking
            printf("Setting up connection for device: %s (Node %d)\n", item->mac_str, item->clientnode);
            
            // Get or create device state (this can involve memory allocation)
            device_state_t* existing_state = get_or_create_device_state(item->mac_str);
            if (existing_state) {
                // Clear any existing buffer state for clean start
                existing_state->data_len = 0;
                existing_state->expecting_fragments = false;
                existing_state->expected_total_samples = 0;
                existing_state->expected_element_size = 0;
                existing_state->last_activity = time(NULL);
                existing_state->fragment_start_time = 0;
                existing_state->has_incomplete_data = false;
                printf("Cleared buffer state for connected device: %s\n", item->mac_str);
            } else {
                printf("Warning: Could not create device state for %s\n", item->mac_str);
            }
            
        } else if (item->type == WORK_TYPE_DISCONNECTION) {
            // Handle disconnection cleanup - moved from callback to prevent blocking
            printf("Cleaning up disconnection for device: %s (Node %d)\n", item->mac_str, item->clientnode);
            
            // Try to salvage incomplete data before releasing device state
            device_state_t* state = get_or_create_device_state(item->mac_str);
            if (state && state->data_len > 0) {
                printf("Attempting to process %zu bytes of incomplete data before disconnect\n", state->data_len);
                char* json_string = process_incomplete_packet(state->buffer, state->data_len,
                                                            item->mac_str, "device_disconnect");
                if (json_string) {
                    send_json_to_python(json_string);
                    free(json_string);
                    printf("Successfully sent incomplete packet data for disconnected device %s\n", item->mac_str);
                } else {
                    printf("Could not process incomplete data for disconnected device %s\n", item->mac_str);
                }
            }
            
            // Release device state
            release_device_state(item->mac_str);
        }
        
        free_work_item(item);
    }
    
    printf("Connection management thread shutting down\n");
    return NULL;
}

// Validates pressure values to filter out corrupted data
static bool is_valid_pressure(float pressure, unsigned int timestamp) {
    // Check for NaN/infinite values first
    if (isnan(pressure) || isinf(pressure)) {
        printf("Rejected pressure: NaN or infinite\n");
        return false;
    }
    
    // Reject extremely small values (like e-36, e-43, e-12 examples from corrupted packets)
    if (fabs(pressure) < 1e-10) {
        printf("Rejected pressure: too small (%.2e)\n", pressure);
        return false;
    }
    
    // Reject values outside reasonable atmospheric pressure range
    if (pressure < 200.0f || pressure > 2000.0f) {
        printf("Rejected pressure: out of range (%.2f hPa)\n", pressure);
        return false;
    }
    
    // Timestamp validation removed - device uses relative timestamps, not Unix time
    
    return true;
}

// Parses the packet and returns a heap-allocated JSON string.
// The caller is responsible for freeing the returned string.
// Returns NULL on failure.
static char* process_and_serialize_packet(const unsigned char* data, size_t len, const char* mac_str) {
    if (len < 37) {  // Minimum size: CMD:(4) + header(28) + payload_header(4) + element_size(1)
        printf("Packet too short: %zu bytes\n", len);
        return NULL;
    }
    
    // Check for CMD: header
    if (data[0] != 'C' || data[1] != 'M' || data[2] != 'D' || data[3] != ':') {
        printf("Invalid packet header\n");
        return NULL;
    }
    
    // Parse the main packet header (starting after CMD:)
    const unsigned char* header_start = data + 4;
    FallyxPacketHeader header;
    
    header.version = *((unsigned int*)&header_start[0]);
    header.id1 = *((unsigned int*)&header_start[4]);
    header.id2 = *((unsigned int*)&header_start[8]);
    header.id3 = *((unsigned int*)&header_start[12]);
    header.tick = *((unsigned int*)&header_start[16]);
    header.config_id = *((unsigned int*)&header_start[20]);
    header.message_id = *((unsigned int*)&header_start[24]);
    header.packet_type = header_start[28];
    
    // Create top-level JSON object
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        printf("Failed to create JSON root object\n");
        return NULL;
    }
    
    // Add MAC address and timestamp
    cJSON_AddStringToObject(root, "mac", mac_str);
    cJSON_AddNumberToObject(root, "received_at", (double)time(NULL));
    
    // Create packet object
    cJSON *packet = cJSON_CreateObject();
    if (!packet) {
        cJSON_Delete(root);
        return NULL;
    }
    
    // Add packet header fields
    cJSON_AddNumberToObject(packet, "version", header.version);
    
    char hw_id_str[32];
    snprintf(hw_id_str, sizeof(hw_id_str), "%08X.%08X.%08X", header.id1, header.id2, header.id3);
    cJSON_AddStringToObject(packet, "hardware_id", hw_id_str);
    
    cJSON_AddNumberToObject(packet, "device_timestamp", header.tick);
    cJSON_AddNumberToObject(packet, "config_id", header.config_id);
    cJSON_AddNumberToObject(packet, "message_id", header.message_id);
    cJSON_AddNumberToObject(packet, "packet_type", header.packet_type);
    
    // Process payload based on packet type
    if (header.packet_type == 0xA1) {
        // Parse IMU data payload - ORIGINAL FORMAT
        unsigned char packet_format = data[33];
        unsigned short num_samples  = *((unsigned short*)&data[34]);
        unsigned char element_size  = data[36];
        
        cJSON *payload = cJSON_CreateObject();
        if (!payload) {
            cJSON_Delete(root);
            return NULL;
        }
        
        cJSON_AddStringToObject(payload, "type", "ACCEL_GYRO");
        cJSON_AddNumberToObject(payload, "format", packet_format);
        cJSON_AddNumberToObject(payload, "data_length", num_samples);
        
        // Parse sensor samples if we have enough data
        int samples_start = 37;  // After CMD:(4) + header(28) + payload_header(4) + element_size(1)
        int expected_payload_size = num_samples * element_size;
        int available_data = len - samples_start;
        
        if (available_data >= expected_payload_size && element_size == 36) {
            cJSON *timestamps = cJSON_CreateArray();
            cJSON *ax = cJSON_CreateArray();
            cJSON *ay = cJSON_CreateArray();
            cJSON *az = cJSON_CreateArray();
            cJSON *gx = cJSON_CreateArray();
            cJSON *gy = cJSON_CreateArray();
            cJSON *gz = cJSON_CreateArray();
            cJSON *aux_data_array = cJSON_CreateArray();
            
            if (!timestamps || !ax || !ay || !az || !gx || !gy || !gz || !aux_data_array) {
                cJSON_Delete(root);
                return NULL;
            }
            
            // Parse each sample
            for (int sample = 0; sample < num_samples && sample < num_samples; sample++) {
                int offset = samples_start + (sample * element_size);
                
                if (offset + 36 > len) {
                    printf("Sample %d: Offset %d + 36 exceeds data length %zu - stopping\n", sample, offset, len);
                    break;
                }
                
                // Extract sensor data
                float accel_x = *((float*)&data[offset + 0]);
                float accel_y = *((float*)&data[offset + 4]);
                float accel_z = *((float*)&data[offset + 8]);
                float gyro_x = *((float*)&data[offset + 12]);
                float gyro_y = *((float*)&data[offset + 16]);
                float gyro_z = *((float*)&data[offset + 20]);
                
                // 8 bytes of auxiliary data
                char aux_hex[17];
                for (int i = 0; i < 8; i++) {
                    sprintf(&aux_hex[i*2], "%02X", data[offset + 24 + i]);
                }
                aux_hex[16] = '\0';
                
                // 4-byte timestamp
                unsigned int timestamp = *((unsigned int*)&data[offset + 32]);
                
                // Add to arrays
                cJSON_AddItemToArray(timestamps, cJSON_CreateNumber(timestamp));
                cJSON_AddItemToArray(ax, cJSON_CreateNumber(accel_x));
                cJSON_AddItemToArray(ay, cJSON_CreateNumber(accel_y));
                cJSON_AddItemToArray(az, cJSON_CreateNumber(accel_z));
                cJSON_AddItemToArray(gx, cJSON_CreateNumber(gyro_x));
                cJSON_AddItemToArray(gy, cJSON_CreateNumber(gyro_y));
                cJSON_AddItemToArray(gz, cJSON_CreateNumber(gyro_z));
                cJSON_AddItemToArray(aux_data_array, cJSON_CreateString(aux_hex));
            }
            
            // Add arrays to payload
            cJSON_AddItemToObject(payload, "Timestamp", timestamps);
            cJSON_AddItemToObject(payload, "Ax", ax);
            cJSON_AddItemToObject(payload, "Ay", ay);
            cJSON_AddItemToObject(payload, "Az", az);
            cJSON_AddItemToObject(payload, "Gx", gx);
            cJSON_AddItemToObject(payload, "Gy", gy);
            cJSON_AddItemToObject(payload, "Gz", gz);
            cJSON_AddItemToObject(payload, "AuxData", aux_data_array);
        }
        
        cJSON_AddItemToObject(packet, "payload", payload);
    }
    else if (header.packet_type == 0x45) {
        // Parse PRESSURE array data - payload header format: packet_format(1) + numSamples(UINT16) + sample_size(UINT8)
        unsigned char packet_format = data[33];                       // Packet format at offset 33
        unsigned short num_samples  = *((unsigned short*)&data[34]);  // UINT16 at offset 34-35
        unsigned char sample_size   = data[36];                       // UINT8 at offset 36
        
        printf("DEBUG 0x45: packet_format at offset 33 = 0x%02X, num_samples = %u, sample_size = %u\n",
               packet_format, num_samples, sample_size);
        
        cJSON *payload = cJSON_CreateObject();
        if (!payload) {
            cJSON_Delete(root);
            return NULL;
        }
        
        cJSON_AddStringToObject(payload, "type", "PRESSURE");
        cJSON_AddNumberToObject(payload, "format", packet_format);
        cJSON_AddNumberToObject(payload, "num_samples", num_samples);
        cJSON_AddNumberToObject(payload, "sample_size", sample_size);
        
        // Parse pressure samples if we have enough data
        int samples_start = 37;  // After CMD:(4) + header(28) + packet_format(1) + numSamples(2) + sample_size(1) + 1 byte padding
        int expected_payload_size = num_samples * sample_size;
        int available_data = len - samples_start;
        
        if (available_data >= expected_payload_size && sample_size == sizeof(PressurePayload)) {
            cJSON *timestamps = cJSON_CreateArray();
            cJSON *pressures = cJSON_CreateArray();
            
            if (!timestamps || !pressures) {
                cJSON_Delete(root);
                return NULL;
            }
            
            // Parse each pressure sample
            for (int sample = 0; sample < num_samples; sample++) {
                int offset = samples_start + (sample * sample_size);
                
                if (offset + sizeof(PressurePayload) > len) {
                    printf("Pressure sample %d: Offset %d + %zu exceeds data length %zu - stopping\n",
                           sample, offset, sizeof(PressurePayload), len);
                    break;
                }
                
                // Extract pressure data according to PressurePayload structure
                unsigned int timestamp = *((unsigned int*)&data[offset + 0]);
                float pressure = *((float*)&data[offset + 4]);
                
                // Validate before adding to JSON - BOTH arrays updated together
                if (is_valid_pressure(pressure, timestamp)) {
                    cJSON_AddItemToArray(timestamps, cJSON_CreateNumber(timestamp));
                    cJSON_AddItemToArray(pressures, cJSON_CreateNumber(pressure));
                } else {
                    printf("Skipping invalid pressure sample %d: pressure=%.2e, timestamp=%u\n",
                           sample, pressure, timestamp);
                    // IMPORTANT: Both timestamp AND pressure are skipped together
                    // This keeps the arrays synchronized
                }
            }
            
            // Add arrays to payload
            cJSON_AddItemToObject(payload, "Timestamp", timestamps);
            cJSON_AddItemToObject(payload, "Pressure", pressures);
        } else {
            printf("Warning: Packet type 0x45 insufficient data: available=%d, expected=%d, sample_size=%d\n",
                   available_data, expected_payload_size, sample_size);
            
            // Add minimal payload info even if data is incomplete
            cJSON_AddStringToObject(payload, "error", "Insufficient data for pressure array");
            cJSON_AddNumberToObject(payload, "available_bytes", available_data);
            cJSON_AddNumberToObject(payload, "expected_bytes", expected_payload_size);
        }
        
        cJSON_AddItemToObject(packet, "payload", payload);
    }
    else if (header.packet_type == 0x41) {
        // Parse TEMP_BATT_GPIO_PAYLOAD data - expects UInt8 size + TempBattGpioPayload
        if (len >= 34 + sizeof(TempBattGpioPayload)) {  // CMD:(4) + header(28) + packet_format(1) + data_size(1) + payload
            unsigned char packet_format = data[33];     // Packet format at offset 33
            unsigned char data_size = data[34];         // UInt8 size at offset 34
            printf("DEBUG 0x41: packet_format at offset 33 = 0x%02X, data_size at offset 34 = %d\n", packet_format, data_size);
            const unsigned char* payload_data = data + 35;  // Skip to payload data after packet_format & data_size
            
            // Validate data_size matches expected TempBattGpioPayload size
            if (data_size != sizeof(TempBattGpioPayload)) {
                printf("Warning: 0x41 data_size mismatch: got %d, expected %zu\n", data_size, sizeof(TempBattGpioPayload));
            }
            
            // Ensure we have enough data for the payload
            int available_payload_data = len - 35;  // Data available after offset 35
            if (available_payload_data < data_size) {
                printf("Warning: 0x41 insufficient payload data: available=%d, needed=%d\n", available_payload_data, data_size);
                
                // Add minimal payload info even if data is incomplete
                cJSON *payload = cJSON_CreateObject();
                if (payload) {
                    cJSON_AddStringToObject(payload, "type", "TEMP_BATT_GPIO");
                    cJSON_AddStringToObject(payload, "error", "Insufficient payload data");
                    cJSON_AddNumberToObject(payload, "data_size", data_size);
                    cJSON_AddNumberToObject(payload, "available_bytes", available_payload_data);
                    cJSON_AddItemToObject(packet, "payload", payload);
                }
                return cJSON_PrintUnformatted(root);
            }
            
            cJSON *payload = cJSON_CreateObject();
            if (!payload) {
                cJSON_Delete(root);
                return NULL;
            }
            
            cJSON_AddStringToObject(payload, "type", "TEMP_BATT_GPIO");
            cJSON_AddNumberToObject(payload, "format", packet_format);
            cJSON_AddNumberToObject(payload, "data_size", data_size);
            
            // Parse the payload data according to TempBattGpioPayload structure
            unsigned int timestamp = *((unsigned int*)&payload_data[0]);
            float voltage = *((float*)&payload_data[4]);
            float current = *((float*)&payload_data[8]);
            float avg_power = *((float*)&payload_data[12]);
            float remaining = *((float*)&payload_data[16]);
            float temperature = *((float*)&payload_data[20]);
            float pressure = *((float*)&payload_data[24]);
            unsigned char fsr1 = payload_data[28];
            unsigned char fsr2 = payload_data[29];
            unsigned char button = payload_data[30];
            unsigned char AccelRate = payload_data[31];
            unsigned char GyroRate = payload_data[32];
            unsigned char AccelDownsample = payload_data[33];
            unsigned char GyroDownsample = payload_data[34];
            
            // Add all fields to JSON payload
            cJSON_AddNumberToObject(payload, "timestamp", timestamp);
            cJSON_AddNumberToObject(payload, "voltage", voltage);
            cJSON_AddNumberToObject(payload, "current", current);
            cJSON_AddNumberToObject(payload, "avg_power", avg_power);
            cJSON_AddNumberToObject(payload, "remaining", remaining);
            cJSON_AddNumberToObject(payload, "temperature", temperature);
            cJSON_AddNumberToObject(payload, "pressure", pressure);
            cJSON_AddNumberToObject(payload, "fsr1", fsr1);
            cJSON_AddNumberToObject(payload, "fsr2", fsr2);
            cJSON_AddNumberToObject(payload, "button", button);
            cJSON_AddNumberToObject(payload, "AccelRate", AccelRate);
            cJSON_AddNumberToObject(payload, "GyroRate", GyroRate);
            cJSON_AddNumberToObject(payload, "AccelDownsample", AccelDownsample);
            cJSON_AddNumberToObject(payload, "GyroDownsample", GyroDownsample);
            
            cJSON_AddItemToObject(packet, "payload", payload);
        } else {
            printf("Warning: Packet type 0x41 too short: %zu bytes (expected %zu bytes)\n", len, 35 + sizeof(TempBattGpioPayload));
            
            // Add minimal payload info even if data is incomplete
            cJSON *payload = cJSON_CreateObject();
            if (payload) {
                cJSON_AddStringToObject(payload, "type", "TEMP_BATT_GPIO");
                cJSON_AddStringToObject(payload, "error", "Incomplete packet data");
                cJSON_AddNumberToObject(payload, "packet_length", len);
                cJSON_AddNumberToObject(payload, "expected_minimum", 35 + sizeof(TempBattGpioPayload));
                cJSON_AddItemToObject(packet, "payload", payload);
            }
        }
    }
    
    // Add packet to root
    cJSON_AddItemToObject(root, "packet", packet);
    
    // Serialize to string
    char* json_string = cJSON_PrintUnformatted(root);
    
    // Clean up
    cJSON_Delete(root);
    
    return json_string;
}

// Process incomplete packet data and return a heap-allocated JSON string.
// This function is more lenient than process_and_serialize_packet and attempts
// to salvage whatever data is available when a device disconnects unexpectedly.
// The caller is responsible for freeing the returned string.
// Returns NULL on failure.
static char* process_incomplete_packet(const unsigned char* data, size_t len, const char* mac_str, const char* reason) {
    if (len < 33) {  // Need at least CMD:(4) + header(28) + packet_type(1)
        printf("Incomplete packet too short for processing: %zu bytes\n", len);
        return NULL;
    }
    
    // Check for CMD: header
    if (data[0] != 'C' || data[1] != 'M' || data[2] != 'D' || data[3] != ':') {
        printf("Incomplete packet has invalid header\n");
        return NULL;
    }
    
    // Parse the main packet header (starting after CMD:)
    const unsigned char* header_start = data + 4;
    FallyxPacketHeader header;
    
    header.version = *((unsigned int*)&header_start[0]);
    header.id1 = *((unsigned int*)&header_start[4]);
    header.id2 = *((unsigned int*)&header_start[8]);
    header.id3 = *((unsigned int*)&header_start[12]);
    header.tick = *((unsigned int*)&header_start[16]);
    header.config_id = *((unsigned int*)&header_start[20]);
    header.message_id = *((unsigned int*)&header_start[24]);
    header.packet_type = header_start[28];
    
    // Create top-level JSON object
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        printf("Failed to create JSON root object for incomplete packet\n");
        return NULL;
    }
    
    // Add MAC address, timestamp, and incomplete metadata
    cJSON_AddStringToObject(root, "mac", mac_str);
    cJSON_AddNumberToObject(root, "received_at", (double)time(NULL));
    cJSON_AddBoolToObject(root, "incomplete", true);
    cJSON_AddStringToObject(root, "incomplete_reason", reason);
    cJSON_AddNumberToObject(root, "actual_data_size", len);
    
    // Add timeout-specific metadata
    if (strstr(reason, "timeout")) {
        cJSON_AddStringToObject(root, "timeout_type",
            strcmp(reason, "fragment_timeout") == 0 ? "fragment" :
            strcmp(reason, "device_timeout") == 0 ? "device" : "unknown");
        cJSON_AddBoolToObject(root, "was_timeout_event", true);
    } else {
        cJSON_AddBoolToObject(root, "was_timeout_event", false);
    }
    
    // Create packet object
    cJSON *packet = cJSON_CreateObject();
    if (!packet) {
        cJSON_Delete(root);
        return NULL;
    }
    
    // Add packet header fields
    cJSON_AddNumberToObject(packet, "version", header.version);
    
    char hw_id_str[32];
    snprintf(hw_id_str, sizeof(hw_id_str), "%08X.%08X.%08X", header.id1, header.id2, header.id3);
    cJSON_AddStringToObject(packet, "hardware_id", hw_id_str);
    
    cJSON_AddNumberToObject(packet, "device_timestamp", header.tick);
    cJSON_AddNumberToObject(packet, "config_id", header.config_id);
    cJSON_AddNumberToObject(packet, "message_id", header.message_id);
    cJSON_AddNumberToObject(packet, "packet_type", header.packet_type);
    
    // Process payload based on packet type - attempt to salvage what we can
    if (header.packet_type == 0xA1 && len >= 37) {
        // Parse IMU data payload - ORIGINAL FORMAT
        unsigned char packet_format = data[33];
        unsigned short num_samples  = *((unsigned short*)&data[34]);
        unsigned char element_size  = data[36];
        
        cJSON *payload = cJSON_CreateObject();
        if (!payload) {
            cJSON_Delete(root);
            return NULL;
        }
        
        cJSON_AddStringToObject(payload, "type", "ACCEL_GYRO");
        cJSON_AddNumberToObject(payload, "format", packet_format);
        cJSON_AddNumberToObject(payload, "expected_samples", num_samples);
        
        // Calculate expected vs actual data
        int samples_start = 37;
        int expected_payload_size = num_samples * element_size;
        int available_data = len - samples_start;
        int actual_samples = (element_size > 0) ? (available_data / element_size) : 0;
        
        cJSON_AddNumberToObject(payload, "expected_total_size", samples_start + expected_payload_size);
        cJSON_AddNumberToObject(payload, "actual_samples", actual_samples);
        
        // Try to parse whatever complete samples we have
        if (available_data > 0 && element_size == 36 && actual_samples > 0) {
            cJSON *timestamps = cJSON_CreateArray();
            cJSON *ax = cJSON_CreateArray();
            cJSON *ay = cJSON_CreateArray();
            cJSON *az = cJSON_CreateArray();
            cJSON *gx = cJSON_CreateArray();
            cJSON *gy = cJSON_CreateArray();
            cJSON *gz = cJSON_CreateArray();
            cJSON *aux_data_array = cJSON_CreateArray();
            
            if (timestamps && ax && ay && az && gx && gy && gz && aux_data_array) {
                // Parse each complete sample we have
                for (int sample = 0; sample < actual_samples && sample < num_samples; sample++) {
                    int offset = samples_start + (sample * element_size);
                    
                    if (offset + 36 > len) {
                        break; // Not enough data for this sample
                    }
                    
                    // Extract sensor data
                    float accel_x = *((float*)&data[offset + 0]);
                    float accel_y = *((float*)&data[offset + 4]);
                    float accel_z = *((float*)&data[offset + 8]);
                    float gyro_x = *((float*)&data[offset + 12]);
                    float gyro_y = *((float*)&data[offset + 16]);
                    float gyro_z = *((float*)&data[offset + 20]);
                    
                    // 8 bytes of auxiliary data
                    char aux_hex[17];
                    for (int i = 0; i < 8; i++) {
                        sprintf(&aux_hex[i*2], "%02X", data[offset + 24 + i]);
                    }
                    aux_hex[16] = '\0';
                    
                    // 4-byte timestamp
                    unsigned int timestamp = *((unsigned int*)&data[offset + 32]);
                    
                    // Add to arrays
                    cJSON_AddItemToArray(timestamps, cJSON_CreateNumber(timestamp));
                    cJSON_AddItemToArray(ax, cJSON_CreateNumber(accel_x));
                    cJSON_AddItemToArray(ay, cJSON_CreateNumber(accel_y));
                    cJSON_AddItemToArray(az, cJSON_CreateNumber(accel_z));
                    cJSON_AddItemToArray(gx, cJSON_CreateNumber(gyro_x));
                    cJSON_AddItemToArray(gy, cJSON_CreateNumber(gyro_y));
                    cJSON_AddItemToArray(gz, cJSON_CreateNumber(gyro_z));
                    cJSON_AddItemToArray(aux_data_array, cJSON_CreateString(aux_hex));
                }
                
                // Add arrays to payload
                cJSON_AddItemToObject(payload, "Timestamp", timestamps);
                cJSON_AddItemToObject(payload, "Ax", ax);
                cJSON_AddItemToObject(payload, "Ay", ay);
                cJSON_AddItemToObject(payload, "Az", az);
                cJSON_AddItemToObject(payload, "Gx", gx);
                cJSON_AddItemToObject(payload, "Gy", gy);
                cJSON_AddItemToObject(payload, "Gz", gz);
                cJSON_AddItemToObject(payload, "AuxData", aux_data_array);
            }
        }
        
        cJSON_AddItemToObject(packet, "payload", payload);
    }
    else if (header.packet_type == 0x45 && len >= 37) {
        // Parse PRESSURE array data
        unsigned char packet_format = data[33];
        unsigned short num_samples  = *((unsigned short*)&data[34]);
        unsigned char sample_size   = data[36];
        
        cJSON *payload = cJSON_CreateObject();
        if (!payload) {
            cJSON_Delete(root);
            return NULL;
        }
        
        cJSON_AddStringToObject(payload, "type", "PRESSURE");
        cJSON_AddNumberToObject(payload, "format", packet_format);
        cJSON_AddNumberToObject(payload, "expected_samples", num_samples);
        cJSON_AddNumberToObject(payload, "sample_size", sample_size);
        
        // Calculate expected vs actual data
        int samples_start = 37;
        int expected_payload_size = num_samples * sample_size;
        int available_data = len - samples_start;
        int actual_samples = (sample_size > 0) ? (available_data / sample_size) : 0;
        
        cJSON_AddNumberToObject(payload, "expected_total_size", samples_start + expected_payload_size);
        cJSON_AddNumberToObject(payload, "actual_samples", actual_samples);
        
        // Try to parse whatever complete samples we have
        if (available_data > 0 && sample_size == sizeof(PressurePayload) && actual_samples > 0) {
            cJSON *timestamps = cJSON_CreateArray();
            cJSON *pressures = cJSON_CreateArray();
            
            if (timestamps && pressures) {
                // Parse each complete pressure sample
                for (int sample = 0; sample < actual_samples; sample++) {
                    int offset = samples_start + (sample * sample_size);
                    
                    if (offset + sizeof(PressurePayload) > len) {
                        break; // Not enough data for this sample
                    }
                    
                    // Extract pressure data
                    unsigned int timestamp = *((unsigned int*)&data[offset + 0]);
                    float pressure = *((float*)&data[offset + 4]);
                    
                    // Validate before adding to JSON - BOTH arrays updated together
                    if (is_valid_pressure(pressure, timestamp)) {
                        cJSON_AddItemToArray(timestamps, cJSON_CreateNumber(timestamp));
                        cJSON_AddItemToArray(pressures, cJSON_CreateNumber(pressure));
                    }
                    // If invalid, both timestamp and pressure are skipped together
                }
                
                // Add arrays to payload
                cJSON_AddItemToObject(payload, "Timestamp", timestamps);
                cJSON_AddItemToObject(payload, "Pressure", pressures);
            }
        }
        
        cJSON_AddItemToObject(packet, "payload", payload);
    }
    else if (header.packet_type == 0x41 && len >= 35) {
        // Parse TEMP_BATT_GPIO_PAYLOAD data
        unsigned char packet_format = data[33];
        unsigned char data_size = (len >= 35) ? data[34] : 0;
        
        cJSON *payload = cJSON_CreateObject();
        if (!payload) {
            cJSON_Delete(root);
            return NULL;
        }
        
        cJSON_AddStringToObject(payload, "type", "TEMP_BATT_GPIO");
        cJSON_AddNumberToObject(payload, "format", packet_format);
        cJSON_AddNumberToObject(payload, "expected_data_size", data_size);
        cJSON_AddNumberToObject(payload, "expected_total_size", 35 + data_size);
        
        // Try to parse whatever data we have if we have at least some payload
        int available_payload_data = len - 35;
        if (available_payload_data > 0) {
            const unsigned char* payload_data = data + 35;
            
            // Parse as much as we can safely
            if (available_payload_data >= 4) {
                unsigned int timestamp = *((unsigned int*)&payload_data[0]);
                cJSON_AddNumberToObject(payload, "timestamp", timestamp);
            }
            if (available_payload_data >= 8) {
                float voltage = *((float*)&payload_data[4]);
                cJSON_AddNumberToObject(payload, "voltage", voltage);
            }
            if (available_payload_data >= 12) {
                float current = *((float*)&payload_data[8]);
                cJSON_AddNumberToObject(payload, "current", current);
            }
            if (available_payload_data >= 16) {
                float avg_power = *((float*)&payload_data[12]);
                cJSON_AddNumberToObject(payload, "avg_power", avg_power);
            }
            if (available_payload_data >= 20) {
                float remaining = *((float*)&payload_data[16]);
                cJSON_AddNumberToObject(payload, "remaining", remaining);
            }
            if (available_payload_data >= 24) {
                float temperature = *((float*)&payload_data[20]);
                cJSON_AddNumberToObject(payload, "temperature", temperature);
            }
            if (available_payload_data >= 28) {
                float pressure = *((float*)&payload_data[24]);
                cJSON_AddNumberToObject(payload, "pressure", pressure);
            }
            if (available_payload_data >= 29) {
                unsigned char fsr1 = payload_data[28];
                cJSON_AddNumberToObject(payload, "fsr1", fsr1);
            }
            if (available_payload_data >= 30) {
                unsigned char fsr2 = payload_data[29];
                cJSON_AddNumberToObject(payload, "fsr2", fsr2);
            }
            if (available_payload_data >= 31) {
                unsigned char button = payload_data[30];
                cJSON_AddNumberToObject(payload, "button", button);
            }
            if (available_payload_data >= 32) {
                unsigned char AccelRate = payload_data[31];
                cJSON_AddNumberToObject(payload, "AccelRate", AccelRate);
            }
            if (available_payload_data >= 33) {
                unsigned char GyroRate = payload_data[32];
                cJSON_AddNumberToObject(payload, "GyroRate", GyroRate);
            }
            if (available_payload_data >= 34) {
                unsigned char AccelDownsample = payload_data[33];
                cJSON_AddNumberToObject(payload, "AccelDownsample", AccelDownsample);
            }
            if (available_payload_data >= 35) {
                unsigned char GyroDownsample = payload_data[34];
                cJSON_AddNumberToObject(payload, "GyroDownsample", GyroDownsample);
            }
            
            cJSON_AddNumberToObject(payload, "available_payload_bytes", available_payload_data);
        }
        
        cJSON_AddItemToObject(packet, "payload", payload);
    }
    else {
        // Unknown or insufficient data for packet type
        cJSON *payload = cJSON_CreateObject();
        if (payload) {
            cJSON_AddStringToObject(payload, "type", "UNKNOWN_OR_INSUFFICIENT");
            cJSON_AddStringToObject(payload, "error", "Insufficient data to parse payload");
            cJSON_AddNumberToObject(payload, "minimum_required", 37);
            cJSON_AddItemToObject(packet, "payload", payload);
        }
    }
    
    // Add packet to root
    cJSON_AddItemToObject(root, "packet", packet);
    
    // Serialize to string
    char* json_string = cJSON_PrintUnformatted(root);
    
    // Clean up
    cJSON_Delete(root);
    
    return json_string;
}

// Modified function to send JSON strings
void send_json_to_python(const char* json_string) {
    if (py_socket_fd == -1) {
        printf("Socket not connected. Attempting to reconnect...\n");
        connect_to_python_socket();
        if (py_socket_fd == -1) {
            printf("Error: Reconnection failed. Cannot send data.\n");
            return;
        }
    }

    // Use MSG_NOSIGNAL to prevent the daemon from crashing on a broken pipe
    if (send(py_socket_fd, json_string, strlen(json_string), MSG_NOSIGNAL) == -1) {
        perror("send error (broken pipe likely)");
        close(py_socket_fd);
        py_socket_fd = -1;
        return;
    }

    // Also use MSG_NOSIGNAL for the newline delimiter
    if (send(py_socket_fd, "\n", 1, MSG_NOSIGNAL) == -1) {
        perror("send newline error (broken pipe likely)");
        close(py_socket_fd);
        py_socket_fd = -1;
        return;
    }

    printf("Sent JSON data to Python (%zu bytes + newline)\n", strlen(json_string));
}

// --- BLE Callback Handler ---
int le_callback_handler(int clientnode, int operation, int cticn, unsigned char* data, int len) {
    static unsigned char dat[8192];  // Local buffer 
    int nread;
    
    switch (operation) {
        case LE_CONNECT:
            printf("Device connected: %s (Node %d) - queuing connection setup\n", device_name(clientnode), clientnode);
            // LIGHTWEIGHT CALLBACK: Just queue connection event for background processing
            char* connect_mac_str = device_address(clientnode);
            if (connect_mac_str) {
                work_item_t* connect_item = create_connection_work_item(clientnode, connect_mac_str, WORK_TYPE_CONNECTION);
                if (connect_item) {
                    queue_connection_item(connect_item);
                    printf("Queued connection setup for %s - callback returning immediately\n", connect_mac_str);
                } else {
                    printf("Error: Failed to create connection work item for device %s\n", connect_mac_str);
                }
            }
            break;
        case LE_DISCONNECT:
            printf("Device disconnected: %s (Node %d) - queuing disconnection cleanup\n", device_name(clientnode), clientnode);
            // LIGHTWEIGHT CALLBACK: Just queue disconnection event for background processing
            char* mac_str = device_address(clientnode);
            if (mac_str) {
                work_item_t* disconnect_item = create_connection_work_item(clientnode, mac_str, WORK_TYPE_DISCONNECTION);
                if (disconnect_item) {
                    queue_connection_item(disconnect_item);
                    printf("Queued disconnection cleanup for %s - callback returning immediately\n", mac_str);
                } else {
                    printf("Error: Failed to create disconnection work item for device %s\n", mac_str);
                }
            }
            break;
        case LE_WRITE:
            if (cticn == data_tx_ctic_index) {
                // Read local characteristic that client has just written
                nread = read_ctic(localnode(), cticn, dat, sizeof(dat));
                printf("  %s has written to [%s] (%d bytes)\n", device_name(clientnode), ctic_name(localnode(), cticn), nread);
                
                if (nread > 0) {
                    // Get device MAC address
                    char* mac_str = device_address(clientnode);
                    if (!mac_str) {
                        printf("Error: Could not get MAC address for node %d\n", clientnode);
                        break;
                    }
                    
                    // LIGHTWEIGHT CALLBACK: Just create work item and queue it
                    // All heavy processing is now done in the background thread
                    work_item_t* item = create_work_item(clientnode, dat, nread, mac_str);
                    if (item) {
                        queue_work_item(item);
                        printf("Queued work item for %s (%d bytes) - callback returning immediately\n", mac_str, nread);
                    } else {
                        printf("Error: Failed to create work item for device %s\n", mac_str);
                    }
                }
            }
            break;
        default:
            break;
    }
    return SERVER_CONTINUE;
}

// --- MAC Address Verification Functions ---
// Get the actual MAC address from hci0 interface using system commands
char* get_hci0_mac_address() {
    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    static char mac_address[18];
    
    // Try hciconfig first
    fp = popen("hciconfig hci0 2>/dev/null | grep 'BD Address' | awk '{print $3}'", "r");
    if (fp != NULL) {
        if (fgets(mac_address, sizeof(mac_address), fp) != NULL) {
            // Remove newline if present
            char *newline = strchr(mac_address, '\n');
            if (newline) *newline = '\0';
            
            // Check if we got a valid MAC address (should be 17 characters: XX:XX:XX:XX:XX:XX)
            if (strlen(mac_address) == 17) {
                pclose(fp);
                printf("Found MAC address via hciconfig: %s\n", mac_address);
                return mac_address;
            }
        }
        pclose(fp);
    }
    
    // Try alternative method with bluetoothctl
    fp = popen("bluetoothctl show 2>/dev/null | grep 'Controller' | awk '{print $2}'", "r");
    if (fp != NULL) {
        if (fgets(mac_address, sizeof(mac_address), fp) != NULL) {
            // Remove newline if present
            char *newline = strchr(mac_address, '\n');
            if (newline) *newline = '\0';
            
            // Check if we got a valid MAC address
            if (strlen(mac_address) == 17) {
                pclose(fp);
                printf("Found MAC address via bluetoothctl: %s\n", mac_address);
                return mac_address;
            }
        }
        pclose(fp);
    }
    
    // Try reading from /sys/class/bluetooth/hci0/address
    fp = fopen("/sys/class/bluetooth/hci0/address", "r");
    if (fp != NULL) {
        if (fgets(mac_address, sizeof(mac_address), fp) != NULL) {
            // Remove newline if present
            char *newline = strchr(mac_address, '\n');
            if (newline) *newline = '\0';
            
            // Check if we got a valid MAC address
            if (strlen(mac_address) == 17) {
                fclose(fp);
                printf("Found MAC address via /sys/class/bluetooth/hci0/address: %s\n", mac_address);
                return mac_address;
            }
        }
        fclose(fp);
    }
    
    printf("Warning: Could not determine hci0 MAC address using any method\n");
    return NULL;
}

// Extract MAC address from existing devices.txt file
char* get_devices_txt_mac_address() {
    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    static char mac_address[18];
    
    fp = fopen("devices.txt", "r");
    if (fp == NULL) {
        printf("Could not open devices.txt for MAC address extraction\n");
        return NULL;
    }
    
    // Look for the ADDRESS= line
    while ((read = getline(&line, &len, fp)) != -1) {
        // Look for ADDRESS= in the line
        char *address_pos = strstr(line, "ADDRESS=");
        if (address_pos != NULL) {
            // Extract the MAC address (should be 17 characters after ADDRESS=)
            char *mac_start = address_pos + 8; // Skip "ADDRESS="
            
            // Copy up to 17 characters or until whitespace/newline
            int i;
            for (i = 0; i < 17 && mac_start[i] != '\0' && mac_start[i] != ' ' &&
                 mac_start[i] != '\t' && mac_start[i] != '\n' && mac_start[i] != '\r'; i++) {
                mac_address[i] = mac_start[i];
            }
            mac_address[i] = '\0';
            
            // Validate MAC address format (should be 17 characters)
            if (strlen(mac_address) == 17) {
                free(line);
                fclose(fp);
                printf("Found MAC address in devices.txt: %s\n", mac_address);
                return mac_address;
            }
        }
    }
    
    free(line);
    fclose(fp);
    printf("Could not find valid MAC address in devices.txt\n");
    return NULL;
}

// Verify that the MAC address in devices.txt matches the actual hci0 MAC address
int verify_mac_address_match() {
    char *actual_mac = get_hci0_mac_address();
    char *devices_mac = get_devices_txt_mac_address();
    
    if (actual_mac == NULL) {
        printf("Warning: Could not determine actual MAC address - skipping verification\n");
        return 1; // Continue anyway
    }
    
    if (devices_mac == NULL) {
        printf("Warning: Could not extract MAC address from devices.txt - will regenerate\n");
        return 0; // Need to regenerate
    }
    
    // Compare MAC addresses (case insensitive)
    if (strcasecmp(actual_mac, devices_mac) == 0) {
        printf("MAC address verification passed: %s\n", actual_mac);
        return 1; // Match - OK to proceed
    } else {
        printf("MAC address mismatch detected!\n");
        printf("  Actual hci0 MAC: %s\n", actual_mac);
        printf("  devices.txt MAC: %s\n", devices_mac);
        printf("  Will regenerate devices.txt with correct MAC address\n");
        return 0; // Mismatch - need to regenerate
    }
}

// --- Auto-Generate devices.txt with correct local address ---
int create_devices_config_file() {
    FILE *file = fopen("devices.txt", "w");
    if (!file) {
        printf("ERROR: Could not create devices.txt file (check permissions and current directory)\n");
        return 0;
    }
    
    // Get the actual MAC address from hci0 interface (no btlib needed)
    char* local_address = get_hci0_mac_address();
    if (!local_address) {
        printf("ERROR: Could not get local Bluetooth address from hci0 interface\n");
        printf("Make sure Bluetooth is enabled and hci0 interface is available\n");
        printf("Try running: sudo hciconfig hci0 up\n");
        fclose(file);
        return 0;
    }
    
    printf("Auto-generating devices.txt with verified local address: %s\n", local_address);
    printf("WARNING: Using default BLE name 'FallyxGateway'\n");
    printf("To customize the BLE name and location, please run the compile script instead:\n");
    printf("  ./compile_gateway.sh\n");
    
    // Write the device configuration with the actual local address and default name
    fprintf(file, "DEVICE=FallyxGateway  TYPE=MESH  NODE=1  CHANNEL=1 ADDRESS=%s\n", local_address);
    fprintf(file, "  PRIMARY_SERVICE = 1800\n");
    fprintf(file, "    LECHAR = Device name  PERMIT=06  SIZE=16  UUID=2A00   ; index 0\n");
    fprintf(file, "  PRIMARY_SERVICE = 55535343fe7d4ae58fa99fafd205e455\n");
    fprintf(file, "    LECHAR = DataTx  PERMIT=06  SIZE=16 UUID=49535343884143f4a8d4ecbe34729bb3        ; index 1\n");
    fprintf(file, "    LECHAR = DataRx  PERMIT=06  SIZE=16 UUID=495353431e4d4bd9ba6123c647249616        ; index 2\n");
    
    fclose(file);
    printf("devices.txt file created successfully with MAC address: %s\n", local_address);
    return 1;
}

void signal_handler(int sig) {
    printf("\nCaught signal %d, shutting down...\n", sig);
    running = 0;
    
    // Force exit if signal is received twice
    static int signal_count = 0;
    signal_count++;
    if (signal_count >= 2) {
        printf("Force exit after multiple signals\n");
        exit(1);
    }
}

// --- Main Function ---
int main(int argc, char *argv[]) {
    // Initialize device pool
    init_device_pool();
    
    // Register signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Check if devices.txt exists and verify MAC address matches hci0 interface
    printf("Checking devices.txt and verifying MAC address...\n");
    
    FILE *check_file = fopen("devices.txt", "r");
    int need_regenerate = 0;
    
    if (!check_file) {
        printf("devices.txt not found. Will auto-generate with local Bluetooth address...\n");
        need_regenerate = 1;
    } else {
        fclose(check_file);
        printf("devices.txt exists. Verifying MAC address matches hci0 interface...\n");
        
        // Verify MAC address matches the actual hci0 interface
        if (verify_mac_address_match() == 0) {
            printf("MAC address mismatch detected. Auto-fixing devices.txt with correct MAC address...\n");
            need_regenerate = 1;
        } else {
            printf("MAC address verification passed. Using existing devices.txt file\n");
        }
    }
    
    // Create or regenerate devices.txt if needed
    if (need_regenerate) {
        printf("Generating/updating devices.txt with correct local Bluetooth address...\n");
        if (create_devices_config_file() == 0) {
            printf("WARNING: Could not create/update devices.txt file\n");
            printf("Make sure you're running the daemon from the ble-daemon directory\n");
            printf("and have write permissions in the current directory.\n");
            printf("Also ensure hci0 Bluetooth interface is available and working.\n");
            printf("Attempting to continue with existing devices.txt (if any)...\n");
            // Don't return -1, try to continue with existing file
        } else {
            printf("devices.txt successfully created/updated with verified MAC address\n");
        }
    }
    
    // Now initialize btlib with the devices.txt file
    if (init_blue("devices.txt") == 0) {
        printf("Failed to initialize btlib. Check devices.txt and sudo permissions. Exiting.\n");
        return -1;
    }
    printf("btlib initialized. Local device name: %s\n", device_name(localnode()));

    // Find the DataTx characteristic index (sensor transmits TO us on this characteristic)
    unsigned char data_tx_uuid[16];
    int num_bytes;
    unsigned char* temp_uuid_ptr = strtohex("49535343884143f4a8d4ecbe34729bb3", &num_bytes);  // DataTx UUID
    if (num_bytes == 16) {
        memcpy(data_tx_uuid, temp_uuid_ptr, 16);
    } else {
        printf("Warning: UUID conversion failed, expected 16 bytes, got %d\n", num_bytes);
        memset(data_tx_uuid, 0, 16);
    }
    
    data_tx_ctic_index = find_ctic_index(localnode(), UUID_16, data_tx_uuid);
    if (data_tx_ctic_index == -1) {
        printf("FATAL: Could not find DataTx characteristic in devices.txt for the local gateway.\n");
        printf("Try deleting devices.txt and running again to regenerate it.\n");
        return -1;
    }
    printf("DataTx characteristic found at index: %d\n", data_tx_ctic_index);

    connect_to_python_socket();
    
    // Start the processing thread before starting BLE server
    printf("Starting background processing thread...\n");
    if (pthread_create(&processing_thread, NULL, processing_thread_func, NULL) != 0) {
        printf("FATAL: Failed to create processing thread\n");
        return -1;
    }
    printf("Processing thread started successfully\n");
    
    // Start the connection management thread
    printf("Starting connection management thread...\n");
    if (pthread_create(&connection_thread, NULL, connection_thread_func, NULL) != 0) {
        printf("FATAL: Failed to create connection management thread\n");
        return -1;
    }
    printf("Connection management thread started successfully\n");
    
    // Start the timeout monitoring thread
    printf("Starting timeout monitoring thread...\n");
    if (pthread_create(&timeout_monitor_thread, NULL, timeout_monitor_thread_func, NULL) != 0) {
        printf("FATAL: Failed to create timeout monitoring thread\n");
        return -1;
    }
    printf("Timeout monitoring thread started successfully\n");
    
    mesh_on();
    printf("BLE advertising started as '%s'\n", device_name(localnode()));
    printf("Starting BLE GATT server, accepting all connections...\n");
    
    // Use the standard le_server call which handles 'x' key press properly
    le_server((void*)le_callback_handler, 0);

    printf("Server stopped. Closing resources.\n");
    
    // Signal all threads to stop and wait for them
    running = 0;
    pthread_cond_signal(&work_queue.condition);      // Wake up processing thread
    pthread_cond_signal(&connection_queue.condition); // Wake up connection thread
    
    printf("Waiting for processing thread to finish...\n");
    pthread_join(processing_thread, NULL);
    printf("Processing thread stopped\n");
    
    printf("Waiting for connection management thread to finish...\n");
    pthread_join(connection_thread, NULL);
    printf("Connection management thread stopped\n");
    
    printf("Waiting for timeout monitoring thread to finish...\n");
    pthread_join(timeout_monitor_thread, NULL);
    printf("Timeout monitoring thread stopped\n");
    
    // Clean up work queues
    cleanup_work_queue();
    cleanup_connection_queue();
    
    mesh_off();
    if (py_socket_fd != -1) {
        close(py_socket_fd);
    }
    close_all();
    return 0;
}

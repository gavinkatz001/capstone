// Fallyx Gateway Example

// includes
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "btlib.h"  

// Global reassembly buffer for fragmented packets
static unsigned char reassembly_buffer[8192];  // Large buffer for complete packets
static int reassembly_length = 0;
static int expecting_fragments = 0;
static unsigned int expected_total_samples = 0;
static unsigned int expected_element_size = 0;

// called on device IO
int le_callback_handler(int clientnode, int operation, int cticn, unsigned char* data, int len)
{
    int n,nread;
    int isCmd = 0;
    static unsigned char dat[8192];  // Increased buffer size to handle large reassembled packets

    if(operation == LE_CONNECT)
        printf("  %s has connected\n", device_name(clientnode));
    else if(operation == LE_READ)
        printf("  %s has read local characteristic %s\n",device_name(clientnode),ctic_name(localnode(),cticn));
    else if(operation == LE_WRITE)
    {
        // read local characteristic that client has just written
        nread = read_ctic(localnode(),cticn,dat,sizeof(dat));
        printf("  %s has written to [%s] (%d bytes)\n",device_name(clientnode),ctic_name(localnode(),cticn), nread);
        
        // Log raw hex data to file for debugging
        FILE *hex_file = fopen("raw_BLE_DATA.txt", "a");
        if (hex_file != NULL) {
            fprintf(hex_file, "=== BLE Data Received ===\n");
            fprintf(hex_file, "Device: %s\n", device_name(clientnode));
            fprintf(hex_file, "Characteristic: %s\n", ctic_name(localnode(),cticn));
            fprintf(hex_file, "Length: %d bytes\n", nread);
            fprintf(hex_file, "Raw Hex: ");
            for(int i = 0; i < nread; i++) {
                fprintf(hex_file, "%02X ", dat[i]);
            }
            fprintf(hex_file, "\n");
            fprintf(hex_file, "Raw ASCII: ");
            for(int i = 0; i < nread; i++) {
                if(dat[i] >= 32 && dat[i] <= 126) {
                    fprintf(hex_file, "%c", dat[i]);
                } else {
                    fprintf(hex_file, ".");
                }
            }
            fprintf(hex_file, "\n");
            fprintf(hex_file, "\n");
            fclose(hex_file);
        }
        
        // Handle packet reassembly for fragmented data
        if (nread >= 4 && dat[0] == 'C' && dat[1] == 'M' && dat[2] == 'D' && dat[3] == ':') {
            // This is the start of a new packet - reset reassembly buffer
            printf("=== NEW PACKET START ===\n");
            reassembly_length = 0;
            expecting_fragments = 0;
            
            // Copy this fragment to reassembly buffer
            if (nread <= sizeof(reassembly_buffer)) {
                memcpy(reassembly_buffer, dat, nread);
                reassembly_length = nread;
                
                // If this fragment has enough data to read the header, extract expected size
                if (nread >= 37) { // CMD:(4) + header(28) + payload_header(4) + element_size(1)
                    unsigned short numSamples = *((unsigned short*)&dat[34]);
                    unsigned char elementSize = dat[36];
                    expected_total_samples = numSamples;
                    expected_element_size = elementSize;
                    
                    int expected_total_size = 37 + (numSamples * elementSize); // header + payload
                    printf("Expected total packet size: %d bytes (%u samples × %u bytes + 37 header)\n",
                           expected_total_size, numSamples, elementSize);
                    
                    if (nread < expected_total_size) {
                        expecting_fragments = 1;
                        printf("Expecting more fragments (have %d of %d bytes)\n", nread, expected_total_size);
                    }
                }
            }
        } else if (expecting_fragments && reassembly_length > 0) {
            // This is a continuation fragment
            printf("=== FRAGMENT RECEIVED ===\n");
            printf("Adding %d bytes to existing %d bytes\n", nread, reassembly_length);
            
            // Append this fragment to reassembly buffer
            if (reassembly_length + nread <= sizeof(reassembly_buffer)) {
                memcpy(reassembly_buffer + reassembly_length, dat, nread);
                reassembly_length += nread;
                
                // Check if we now have a complete packet
                int expected_total_size = 37 + (expected_total_samples * expected_element_size);
                printf("Reassembly buffer now has %d bytes (need %d)\n", reassembly_length, expected_total_size);
                
                if (reassembly_length >= expected_total_size) {
                    printf("=== COMPLETE PACKET ASSEMBLED ===\n");
                    expecting_fragments = 0;
                    
                    // Process the complete packet
                    memcpy(dat, reassembly_buffer, reassembly_length);
                    nread = reassembly_length;
                    
                    // Continue with normal processing below
                } else {
                    // Still waiting for more fragments
                    printf("Still waiting for more fragments\n");
                    return SERVER_CONTINUE;
                }
            }
        }
	
        if (nread >= 4)
	{
		if (dat[0] == 'C')
		if (dat[1] == 'M')
		if (dat[2] == 'D')
		if (dat[3] == ':')
		{
			isCmd = 1;
			//printf("Found CMD:\n");
		}
		if (isCmd == 1)
		if (nread >= 34)
		{
		    unsigned int versionCode = *((unsigned int*)&dat[4]);
                    unsigned int Id1 = *((unsigned int*)&dat[8]);
                    unsigned int Id2 = *((unsigned int*)&dat[12]);
                    unsigned int Id3 = *((unsigned int*)&dat[16]);
		    unsigned int tick = *((unsigned int*)&dat[20]);
                    unsigned int configId = *((unsigned int*)&dat[24]);
                    unsigned int messageId = *((unsigned int*)&dat[28]);
		    unsigned char packetType = dat[32];
		   printf("COMMAND:\n");
	           printf("   Version    = %u\n", versionCode);
                   printf("   HardwareId = %08X.%08X.%08X\n", Id1, Id2, Id3);
                   printf("   Tick       = %u\n", tick);
                   printf("   ConfigId   = %u\n", configId);
                   printf("   MessageId  = %u\n", messageId);
                   printf("   PacketType = 0x%02X\n", packetType);

	           if (packetType == 0xA1)
	                  {
	                      printf("PacketType = ACCEL_GRYO\n");
	                      unsigned char packetFormat = dat[33];
	                      printf("    Format     = 0x%02X\n", packetFormat);
	                      unsigned short numSamples  = *((unsigned short*)&dat[34]);
	                      unsigned char elementSize  = dat[36];

	                      printf("    NumSamples = %u\n",  numSamples);
	                      printf("    ElementSize = %u bytes\n", elementSize);
	                      
	                      // Decode sensor data and write to file
	                      FILE *decode_file = fopen("decoded_sensor_data.txt", "a");
	                      if (decode_file != NULL) {
	                          fprintf(decode_file, "=== Decoded Sensor Data ===\n");
	                          fprintf(decode_file, "Device: %s\n", device_name(clientnode));
	                          fprintf(decode_file, "Version: %u\n", versionCode);
	                          fprintf(decode_file, "HardwareId: %08X.%08X.%08X\n", Id1, Id2, Id3);
	                          fprintf(decode_file, "Tick: %u\n", tick);
	                          fprintf(decode_file, "ConfigId: %u\n", configId);
	                          fprintf(decode_file, "MessageId: %u\n", messageId);
	                          fprintf(decode_file, "PacketType: 0x%02X\n", packetType);
	                          fprintf(decode_file, "Format: 0x%02X\n", packetFormat);
	                          fprintf(decode_file, "NumSamples: %u\n", numSamples);
	                          fprintf(decode_file, "ElementSize: %u bytes\n", elementSize);
	                          fprintf(decode_file, "\n");
	                          
	                          // Write raw hex data
	                          fprintf(decode_file, "Raw Hex Data (%d bytes):\n", nread);
	                          for(int i = 0; i < nread; i++) {
	                              if(i % 16 == 0) fprintf(decode_file, "%08X: ", i);
	                              fprintf(decode_file, "%02X ", dat[i]);
	                              if(i % 16 == 15) fprintf(decode_file, "\n");
	                          }
	                          if(nread % 16 != 0) fprintf(decode_file, "\n");
	                          fprintf(decode_file, "\n");
	                          
	                          // Decode sensor samples if we have enough data
	                          int payload_start = 37; // After CMD:(4) + header(28) + payload_header(4) + element_size(1)
	                          int expected_payload_size = numSamples * elementSize;
	                          int available_data = nread - payload_start;
	                          
	                          fprintf(decode_file, "Payload Analysis:\n");
	                          fprintf(decode_file, "  Payload starts at offset: %d\n", payload_start);
	                          fprintf(decode_file, "  Expected payload size: %d bytes (%u samples × %u bytes)\n",
	                                  expected_payload_size, numSamples, elementSize);
	                          fprintf(decode_file, "  Available data: %d bytes\n", available_data);
	                          
	                          if (available_data >= expected_payload_size && elementSize == 36) {
	                              fprintf(decode_file, "\nDecoded Sensor Samples:\n");
	                              fprintf(decode_file, "Sample#  |    Accel X    |    Accel Y    |    Accel Z    |    Gyro X     |    Gyro Y     |    Gyro Z     | Timestamp |   AuxData\n");
	                              fprintf(decode_file, "---------|---------------|---------------|---------------|---------------|---------------|---------------|-----------|------------------\n");
	                              
	                              for(int sample = 0; sample < numSamples && sample < 85; sample++) { // Limit to first 10 samples
	                                  int offset = payload_start + (sample * elementSize);
	                                  
	                                  // Bounds check to prevent bus error
	                                  if (offset + 36 > nread) {
	                                      fprintf(decode_file, "Sample %d: Offset %d + 36 exceeds data length %d - skipping\n", sample, offset, nread);
	                                      break;
	                                  }
	                                  
	                                  // Extract sensor data (assuming little-endian format)
	                                  // 6 floats (accel + gyro) + 8 bytes aux + 4 bytes timestamp
	                                  float accel_x = *((float*)&dat[offset + 0]);
	                                  float accel_y = *((float*)&dat[offset + 4]);
	                                  float accel_z = *((float*)&dat[offset + 8]);
	                                  float gyro_x = *((float*)&dat[offset + 12]);
	                                  float gyro_y = *((float*)&dat[offset + 16]);
	                                  float gyro_z = *((float*)&dat[offset + 20]);
	                                  
	                                  // 8 bytes of auxiliary data
	                                  unsigned char aux_data[8];
	                                  for(int i = 0; i < 8; i++) {
	                                      aux_data[i] = dat[offset + 24 + i];
	                                  }
	                                  
	                                  // 4-byte timestamp
	                                  unsigned int timestamp = *((unsigned int*)&dat[offset + 32]);
	                                  
	                                  fprintf(decode_file, "%7d  | %13.6f | %13.6f | %13.6f | %13.6f | %13.6f | %13.6f | %9u | ",
	                                          sample, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, timestamp);
	                                  
	                                  // Print aux data as hex
	                                  for(int i = 0; i < 8; i++) {
	                                      fprintf(decode_file, "%02X", aux_data[i]);
	                                  }
	                                  fprintf(decode_file, "\n");
	                              }
	                              
	                              if(numSamples > 10) {
	                                  fprintf(decode_file, "... (showing first 85 of %u samples)\n", numSamples);
	                              }
	                          } else {
	                              fprintf(decode_file, "\nCannot decode samples: ");
	                              if(available_data < expected_payload_size) {
	                                  fprintf(decode_file, "insufficient data\n");
	                              } else if(elementSize != 36) {
	                                  fprintf(decode_file, "unexpected element size (expected 36, got %u)\n", elementSize);
	                              }
	                          }
	                          
	                          fprintf(decode_file, "\n=== End Decoded Data ===\n\n");
	                          fclose(decode_file);
	                      }
	                  }

               }

	}


	for(n = 0 ; n < nread ; ++n)
	{
	   //if (n >= 0 && n <= 4)
           //     printf("%c",dat[n]);
	   //else
//                printf("0x%02x ", dat[n]);
	}
        printf("\n");
    }
    else if(operation == LE_DISCONNECT)
    {
        printf("  %s has disconnected - waiting for another connection\n",device_name(clientnode));
        // uncomment next line to stop LE server when client disconnects
        // return(SERVER_EXIT);
        // otherwise LE server will continue and wait for another connection
        // or opeeration from clients that are still connected
    }
    else if(operation == LE_TIMER)
    {
        printf("  Timer\n");
    }
    else if(operation == LE_KEYPRESS)
    {   // cticn is key code
        printf("   Key code = %d\n",cticn);
    }    
    return SERVER_CONTINUE;
}

// classic server callback
int classic_callback(int node,unsigned char *data,int len)
{
    //static unsigned char *xmessage = {"Hello world\n"};
    //static unsigned char *message = {"Send Hello to exit\n"};

    printf("Received: %s", data);
    //if(data[0] == 'H')
    //{
    //    write_node(node,xmessage,strlen(xmessage));
    //    printf("Disconnecting...\n");
    //    return(SERVER_EXIT);
    //}
    //write_node(node,message,strlen(message));
    return SERVER_CONTINUE;
}

// start
int main(int argc, char *argv[])
{ 
    // init library
    if(init_blue("devices.txt") == 0)      
        return(0); 
    unsigned char *mydata = {"Hello world"};
    write_ctic(localnode(), 1, mydata, strlen(mydata));  
    le_server(le_callback_handler, 0);  // timerds=0
    close_all();
    return (0);

    // init library
 /*   if(init_blue("devices.txt") == 0)
        return 0;

    // set flags
    printf("IF FAILS - Experiment with security = 0/1/2/3\n");
    int security = 2;
    int keyflag = KEY_ON | PASSKEY_LOCAL;
    if(security == 1)
        keyflag = KEY_OFF | PASSKEY_LOCAL;
    else if(security == 2)
        keyflag = KEY_OFF | PASSKEY_OFF;
    else if(security == 3)
        keyflag = KEY_ON | PASSKEY_OFF;

    // start server
    //register_serial(strtohex("FCF05AFD-67D8-4F41-83F5-7BEE22C03CDB", NULL), "My custom serial");
    classic_server(ANY_DEVICE, classic_callback, '\n', keyflag);
    classic_server(ANY_DEVICE, classic_callback, '\n', keyflag);
    close_all();
    return 0;*/
}


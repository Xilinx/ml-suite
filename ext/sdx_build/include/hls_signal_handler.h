#include <signal.h>
#include <string>
#include <iostream>

enum CodeStateE {ENTER_WRAPC, DUMP_INPUTS, CALL_C_DUT, DUMP_OUTPUTS, DELETE_CHAR_BUFFERS, ENTER_WRAPC_PC} CodeState;
string CodeStateS[] = {"ENTER_WRAPC", "DUMP_INPUTS", "CALL_C_DUT", "DUMP_OUTPUTS", "DELETE_CHAR_BUFFERS", "ENTER_WRAPC_PC"};

void message_handler (int sig) 
{
    if (sig == SIGFPE)
    {
        cout << "ERROR: System recieved a signal named SIGFPE and the program has to stop immediately!" << endl
        << "This signal was generated due to a fatal arithmetic error." << endl
        << "Possible cause of this problem may be: division by zero, overflow etc." << endl
        << "Current execution stopped during CodeState = " << CodeStateS[CodeState] << "." << endl
        << "You can search CodeState variable name in apatb*.cpp file under ./sim/wrapc dir to locate the position." << endl
        << endl;
    }
    if (sig == SIGSEGV)
    {
        cout << "ERROR: System recieved a signal named SIGSEGV and the program has to stop immediately!" << endl
        << "This signal was generated when a program tries to read or write outside the memory that is allocated for it, or to write memory that can only be read." << endl
        << "Possible cause of this problem may be: 1) the depth setting of pointer type argument is much larger than it needed; 2)insufficient depth of array argument; 3)null pointer etc." << endl
        << "Current execution stopped during CodeState = " << CodeStateS[CodeState] << "." << endl
        << "You can search CodeState variable name in apatb*.cpp file under ./sim/wrapc dir to locate the position." << endl
        << endl;
    }
    if (sig == SIGTERM)
    {
        cout << "ERROR: System recieved a signal named SIGTERM and the program has to stop immediately!" << endl
        << "This signal was caused by the shell command kill." << endl
        << "Current execution stopped during CodeState = " << CodeStateS[CodeState] << "." << endl
        << "You can search CodeState variable name in apatb*.cpp file under ./sim/wrapc dir to locate the position." << endl
        << endl;
    }
    if (sig == SIGINT)
    {
        cout << "ERROR: System recieved a signal named SIGINT and the program has to stop immediately!" << endl
        << "This signal was generated when the user types the INTR character (normally C-c)." << endl
        << "Current execution stopped during CodeState = " << CodeStateS[CodeState] << "." << endl
        << "You can search CodeState variable name in apatb*.cpp file under ./sim/wrapc dir to locate the position." << endl
        << endl;
    }
//#ifdef _MINGW32_
//#else
//    if (sig == SIGKILL)
//    {
//        cout << "ERROR: System recieved a signal named SIGKILL and the program has to stop immediately!" << endl
//        << "The system generated SIGKILL for a process itself under some unusual conditions where the program cannot possibly continue to run." << endl
//        << "Current execution stopped during CodeState = " << CodeStateS[CodeState] << "." << endl
//        << "You can search CodeState variable name in apatb*.cpp file under ./sim/wrapc dir to locate the position." << endl
//        << endl;
//    }
//    if (sig == SIGQUIT)
//    {
//        cout << "ERROR: System recieved a signal named SIGQUIT and the program has to stop immediately!" << endl
//        << "This signal was generated when the user types the QUIT character and produces a core dump." << endl
//        << "Current execution stopped during CodeState = " << CodeStateS[CodeState] << "." << endl
//        << "You can search CodeState variable name in apatb*.cpp file under ./sim/wrapc dir to locate the position." << endl
//        << endl;
//    }
//#endif
    signal(sig, SIG_DFL);
}

void refine_signal_handler () 
{
    signal(SIGFPE, message_handler);
    signal(SIGSEGV, message_handler);
    signal(SIGTERM, message_handler);
    signal(SIGINT, message_handler);
//#ifdef _MINGW32_
//#else
//    signal(SIGKILL, message_handler);
//    signal(SIGQUIT, message_handler);
//#endif
}

import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:google_fonts/google_fonts.dart';

// Import the required widgets for routing
import 'Alfa.dart'; // Import LearningAlfatihahFullWidget
import 'Alfa1.dart'; // Import LearningAlfatihah1Widget

class LearningAlfatihahHomeWidget extends StatefulWidget {
  const LearningAlfatihahHomeWidget({super.key});

  static String routeName = 'LearningAlfatihahome';
  static String routePath = '/learningAlfatihahome';

  @override
  State<LearningAlfatihahHomeWidget> createState() =>
      _LearningAlfatihahHomeWidgetState();
}

class _LearningAlfatihahHomeWidgetState
    extends State<LearningAlfatihahHomeWidget> {
  int selectedIndex = 1;

  // Function to handle bottom navigation
  void onTabTapped(int index) {
    setState(() {
      selectedIndex = index;
    });
  }

  final scaffoldKey = GlobalKey<ScaffoldState>();

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        FocusScope.of(context).unfocus();
        FocusManager.instance.primaryFocus?.unfocus();
      },
      child: Scaffold(
        key: scaffoldKey,
        backgroundColor: Color(0xFFFAFDCB),
        body: SafeArea(
          top: true,
          child: SingleChildScrollView(
            child: Column(
              children: [
                Container(
                  width: MediaQuery.sizeOf(context).width * 3.9,
                  height: MediaQuery.sizeOf(context).height * 8.44,
                  decoration: BoxDecoration(
                    color: Color(0xFFFAFDCB),
                  ),
                  child: Column(
                    children: [
                      Padding(
                        padding: EdgeInsetsDirectional.fromSTEB(15, 15, 15, 0),
                        child: Row(
                          mainAxisSize: MainAxisSize.max,
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: [
                            Padding(
                              padding: EdgeInsetsDirectional.fromSTEB(1, 0, 0, 0),
                              child: IconButton(
                                icon: Icon(
                                  Icons.arrow_back_ios_rounded,
                                  color: Colors.black,
                                  size: 30,
                                ),
                                onPressed: () {
                                  // Navigate to the previous screen
                                  Navigator.pop(context); // Go back on press
                                },
                              ),
                            ),
                            Align(
                              alignment: AlignmentDirectional(0, 0),
                              child: Padding(
                                padding: EdgeInsetsDirectional.fromSTEB(240, 0, 0, 0),
                                child: IconButton(
                                  icon: FaIcon(
                                    FontAwesomeIcons.volumeUp,
                                    color: Colors.black,
                                    size: 30,
                                  ),
                                  onPressed: () {
                                    print('IconButton pressed ...');
                                  },
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      Padding(
                        padding: EdgeInsetsDirectional.fromSTEB(0, 10, 0, 0),
                        child: Column(
                          mainAxisSize: MainAxisSize.max,
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            Text(
                              'Level 5 : Belajar Membaca\nSurah dengan Tajwid  ',
                              textAlign: TextAlign.center,
                              style: GoogleFonts.inter(
                                fontSize: 20,
                                letterSpacing: 0.0,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Align(
                        alignment: AlignmentDirectional(0, -1),
                        child: Padding(
                          padding: EdgeInsetsDirectional.fromSTEB(0, 80, 0, 0),
                          child: Column(
                            mainAxisSize: MainAxisSize.max,
                            children: [
                              Padding(
                                padding: EdgeInsetsDirectional.fromSTEB(8, 0, 8, 0),
                                child: ElevatedButton(
                                  onPressed: () {
                                    // Navigate to LearningAlfatihahFullWidget
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                          builder: (context) => LearningAlfatihahfullWidget()
                                      ),
                                    );
                                  },
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Icon(
                                        Icons.library_books_sharp,
                                        size: 15,
                                      ),
                                      Text('Surah Al - Fatihah'),
                                    ],
                                  ),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Color(0xFFF2CE31),
                                    padding: EdgeInsets.symmetric(horizontal: 16),
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(12),
                                    ),
                                    textStyle: GoogleFonts.inter(
                                      fontWeight: FontWeight.w600,
                                      color: Colors.black,
                                      fontSize: 18,
                                    ),
                                  ),
                                ),
                              ),
                              Padding(
                                padding: EdgeInsetsDirectional.fromSTEB(8, 20, 8, 0),
                                child: ElevatedButton(
                                  onPressed: () {
                                    // Navigate to LearningAlfatihah1Widget
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                          builder: (context) => LearningAlfatihah1Widget()
                                      ),
                                    );
                                  },
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Icon(
                                        Icons.library_books_sharp,
                                        size: 15,
                                      ),
                                      Text('Belajar Membaca Surah Al - Fatihah'),
                                    ],
                                  ),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Color(0xFFF2CE31),
                                    padding: EdgeInsets.symmetric(horizontal: 16),
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(12),
                                    ),
                                    textStyle: GoogleFonts.inter(
                                      fontWeight: FontWeight.w600,
                                      color: Colors.black,
                                      fontSize: 18,
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

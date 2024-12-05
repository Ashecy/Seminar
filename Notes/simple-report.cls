\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{simple-report}[2024/08/11 Simple report Class]

% Load the base class
\LoadClass{ctexart}

% Set up essential packages
\RequirePackage{amsmath, amssymb, amsthm, amsfonts}  % Math packages
\RequirePackage{graphicx}  % Including graphics
\RequirePackage[colorlinks]{hyperref}  % Hyperlinks in PDF
\RequirePackage{titlesec}  % Custom section titles
\RequirePackage{fancyhdr}  % Custom headers and footers
\RequirePackage{color}     % Color support
\RequirePackage{babel}     % Language support
\RequirePackage{lmodern}   % Modern fonts
\RequirePackage{romannum}  % Roman numerals support
\RequirePackage{ccicons}   % Creative Commons licenses
\RequirePackage{braket}    % Inner product notation
\RequirePackage{float}     % Improved floating environments
\RequirePackage{framed}    % Framed boxes
%\RequirePackage{tikz}      % TikZ for drawing
\RequirePackage{subcaption}    % Subfigures
%\RequirePackage{tocloft}   % Customize TOC appearance

% Set page layout
\RequirePackage[
  left=1in,
  right=1in,
  top=1in,
  bottom=1in,
]{geometry}

% Define theorem environments
\theoremstyle{plain}
\newtheorem{theorem}{定理}[section]
\newtheorem{lemma}{引理}[section]
\newtheorem{proposition}{命题}[section]
\newtheorem*{corollary}{推论}

\theoremstyle{definition}
\newtheorem{definition}{定义}[section]
\newtheorem{conjecture}{猜想}[section]
\newtheorem{example}{例}[section]

\theoremstyle{remark}
\newtheorem*{remark}{\normalfont\bfseries 评论}
\newtheorem*{note}{\normalfont\bfseries 注}
\newtheorem{case}{\normalfont\bfseries 案例}

\renewcommand*{\proofname}{\normalfont\bfseries 证明}

%\setlength\parindent{2\ccwd}
\renewcommand{\theequation}{\thesection.\arabic{equation}}

% Customize TOC appearance
%\renewcommand{\cftchapfont}{\bfseries}
%\renewcommand{\cftchappagefont}{\bfseries}
%\setcounter{secnumdepth}{4}  % Enable four-level numbering

% Set default figure/table placement parameters
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.15}

% End of class
\endinput


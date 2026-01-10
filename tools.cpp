#include <petscsnes.h>
#include <petscdmda.h>
//#include <petscpcmg.h>
//#include <petscdmmg.h>
#include "def.h"
#include "stdlib.h"

/*******************************************************************************************
* YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
* CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
********************************************************************************************/
#undef __FUNCT__
#define __FUNCT__ "DataSaveASCII"
PetscErrorCode DataSaveASCII(Vec x, char *filename)
{
	PetscErrorCode ierr;
	PetscViewer    viewer;

	PetscFunctionBegin;
	//ierr = PetscViewerHDF5Open(comm,filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
	ierr = PetscViewerASCIIOpen(comm,filename,&viewer); CHKERRQ(ierr);
	// format: PETSC_VIEWER_ASCII_SYMMODU, PETSC_VIEWER_ASCII_MATLAB, PETSC_VIEWER_ASCII_COMMON
	ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_SYMMODU); CHKERRQ(ierr);
	ierr = VecView(x,viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

/*******************************************************************************************
* YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
* CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
********************************************************************************************/
#undef __FUNCT__
#define __FUNCT__ "DataSaveBin"
PetscErrorCode DataSaveBin(Vec x, char *filename)
{
	PetscErrorCode ierr;
	PetscViewer    viewer;

	PetscFunctionBegin;

	ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
	ierr = VecView(x,viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}


#if 1
#undef __FUNCT__
#define __FUNCT__ "DataSaveVTK"
PetscErrorCode DataSaveVTK(Vec x, char *filename)
{
        PetscErrorCode ierr;
	PetscViewer    viewer;

	PetscFunctionBegin;

	ierr = PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
	ierr = VecView(x,viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
#endif


/*******************************************************************************************
* YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
* CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
********************************************************************************************/
#undef __FUNCT__
#define __FUNCT__ "DataLoadBin"
PetscErrorCode DataLoadBin(Vec x, char *filename)
{
	PetscErrorCode ierr;
	PetscViewer    viewer;

	PetscFunctionBegin;

	ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&viewer); CHKERRQ(ierr);
	ierr = VecLoad(x,viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
